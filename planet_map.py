from __future__ import division, print_function
import numpy as np, sys, os
from enlib import utils
with utils.nowarn(): import h5py
from enlib import config, pmat, mpi, errors, gapfill, enmap, bench
from enlib import fft, array_ops
from enact import filedb, actscan, actdata, cuts, nmat_measure

# NOTE
fft.set_engine('fftw')

config.set("pmat_cut_type",  "full")

parser = config.ArgumentParser(os.environ["HOME"]+"./enkirc")
parser.add_argument("planet")
parser.add_argument("area")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("tag", nargs="?")
parser.add_argument("-R", "--dist",    type=float, default=0.2)
parser.add_argument("-e", "--equator", action="count", default=0)
parser.add_argument("-z", "--zenith",  action="count", default=0)
parser.add_argument("-c", "--cont",    action="store_true")
parser.add_argument("--sim",	       type=str,   default=None, help="Passing a sel here sets up simulation mode. The simulations will consist of data from the sim sel TODs with the scanning pattern of the real TODs, and with the signal read off from the area map")
parser.add_argument("--noiseless",	action="store_true", help="Replace signal with simulation instead of adding them. This can be used to get noise free transfer functions")
parser.add_argument("--dbox",	       type=str,   default=None, help="Select only detectors in y1:y2,x1:x2 in the focalplane, relative to the center of the array, in degrees.")
parser.add_argument("--tags",	       type=str,   default=None)
parser.add_argument("--pol-family",    type=str,   default=None, help="Select only 'A' or 'B' detectors.")
parser.add_argument("--no-abscal",     action="store_true", help="Do not apply the per-TOD absolute calibration. The output maps will be uncalibrated (i.e. in units pW because the relative calibration is still applied). Used as input for the abscal pipeline.")
parser.add_argument("--no-pol",	    action="store_true", help="Only solve for total intensity")
parser.add_argument("--detset",	type=str, help="Absolute path to detset file. Should be .txt file with detector ids as first column and {0, 1} flag in second column.")
parser.add_argument("--no-div",	    action="store_true", help="Do not store div maps.")
parser.add_argument("--no-rhs",	    action="store_true", help="Do not store rhs maps.")
parser.add_argument("--equ",	    action="store_true", help="Map in coordinate system with cross-linking.")
args = parser.parse_args()

zenith = args.zenith - args.equator

comm = mpi.COMM_WORLD
filedb.init()
ids  = filedb.scans[args.sel]
R    = args.dist * utils.degree
csize= 100

dtype= np.float64
ncomp = 3
area = enmap.read_map(filedb.get_patch_path(args.area)).astype(dtype)
shape= area.shape[-2:]
model_fknee = 10
model_alpha = 10
if args.equ:
	sys = "equ:"+args.planet
else:
	sys = "hor:"+args.planet
if not zenith: sys += "/0_0"
utils.mkdir(args.odir)
prefix = args.odir + "/"
if args.tag:  prefix += args.tag + "_"
if args.dbox: dbox = np.array([[float(w) for w in tok.split(":")] for tok in args.dbox.split(",")]).T*utils.degree
else: dbox = None

if args.sim:
	sim_ids = filedb.scans[args.sim][:len(ids)]
	if area.ndim == 2:
		tmp = enmap.zeros((ncomp,)+shape, area.wcs, dtype)
		tmp[0] = area
		area = tmp

def smooth(tod, srate):
	ft   = fft.rfft(tod)
	freq = fft.rfftfreq(tod.shape[-1])*srate
	flt  = 1/(1+(freq/model_fknee)**model_alpha)
	ft  *= flt
	fft.ifft(ft, tod, normalize=True)
	return tod

def broaden_beam_hor(tod, d, ibeam, obeam):
	ft    = fft.rfft(tod)
	k     = 2*np.pi*fft.rfftfreq(d.nsamp, 1/d.srate)
	el    = np.mean(d.boresight[2,::100])
	skyspeed = d.speed*np.cos(el)
	sigma = (obeam**2-ibeam**2)**0.5
	ft *= np.exp(-0.5*(sigma/skyspeed)**2*k**2)
	fft.ifft(ft, tod, normalize=True)

def undo_abscal(data, entry, map, div, rhs):
	''' Divide out the abscal from the input maps (inplace)'''
	abscal = data.gain_correction[entry.tag]
	if data.gain_mode == 'mce':
		abscal /= data.mce_gain
	elif data.gain_mode == 'mce_compat':
		abscal /= data.mce_gain * 1217.8583043
	else:
		raise ValueError('gain_mode {} not understood'.format(data.gain_mode))
	map /= abscal
	rhs /= abscal
	div /= (abscal ** 2)

def select_pol_family(data, pol_family):
	'''
	Only include detectors of certain polarization family.

	Parameters
	----------
	data : Dataset object
	    Uncalibrated dataset.
	pol_family : str
	    Only use detectors from this family, e.g. A or B.
	'''
	array_name = actdata.get_array_name(data.entry.id) # e.g. "pa7".
	good_dets = data.array_info.info['det_uid'][data.array_info.info['pol_family'] == pol_family]
	# Format like "paXX_0001" to match data.dets.
	good_dets = np.asarray([array_name + '_{:04d}'.format(det) for det in good_dets])
	good_dets = np.setdiff1d(data.dets, good_dets, assume_unique=True)
	data.restrict(dets=good_dets)

def read_detset(fname, entry):
	'''
	Read and process detectorset file.

	Arguments
	---------
	fname : str
	    Absolute path to detset file. Should be .txt file with
	    detector ids as first column and {0, 1} flag in second column.
	entry : pixell.bunch.Bunch object
	    TOD entry.

	Returns
	-------
	dets : (ndet) list
	    List with detectors ids, e.g. ['pa5_12', 'pa5_253'].
	'''
	pa = entry.id[-1]
	det_ids = np.loadtxt(fname, usecols=0, dtype=int)
	flags = np.loadtxt(fname, usecols=1, dtype=bool)
	return [f'pa{pa}_{det_id}' for det_id in det_ids[flags]]

for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	bid   = id.replace(":","_")
	entry = filedb.data[id]
	if args.tags: entry.tag = args.tags
	oname = "%s%s_map.fits" % (prefix, bid)
	if args.cont and os.path.isfile(oname):
		print("Skipping %s (already done)" % (id))
		continue
	# Read the tod as usual
	dets = read_detset(args.detset, entry) if args.detset else None
	try:
		if not args.sim:
			with bench.show("read"):
				d = actdata.read(entry, dets=dets)
		else:
			sim_id	  = sim_ids[ind]
			sim_entry = filedb.data[sim_id]
			with bench.show("read"):
				d  = actdata.read(entry, ["boresight"], dets=dets)
				d += actdata.read(sim_entry, exclude=["boresight"], dets=dets)
		if args.pol_family is not None:
			with bench.show(f"Select pol_family : {args.pol_family}"):
				ndet_tmp = d.ndet
				select_pol_family(d, args.pol_family)
		with bench.show("calibrate"):
			d = actdata.calibrate(d, exclude=["autocut"])
		if d.ndet == 0 or d.nsamp < 2: raise errors.DataMissing("no data in tod")
		# Select detectors if needed
		if dbox is not None:
			mid  = np.mean(utils.minmax(d.point_template, 0), 0)
			off  = d.point_template-mid
			good = np.all((off > dbox[0])&(off < dbox[1]),-1)
			d    = d.restrict(dets=d.dets[good])
	except errors.DataMissing as e:
		print("Skipping %s (%s)" % (id, str(e)))
		continue
	print("Processing %s" % id, d.ndet, d.nsamp)
	# Very simple white noise model. This breaks if the beam has been tod-smoothed by this point.
	with bench.show("ivar"):
		tod  = d.tod
		del d.tod
		tod -= np.mean(tod,1)[:,None]
		tod  = tod.astype(dtype)
		diff = tod[:,1:]-tod[:,:-1]
		diff = diff[:,:diff.shape[-1]//csize*csize].reshape(d.ndet,-1,csize)
		ivar = 1/(np.median(np.mean(diff**2,-1),-1)/2**0.5)
		del diff
	# Estimate noise level
	asens = np.sum(ivar)**-0.5 / d.srate**0.5
	with bench.show("actscan"):
		scan = actscan.ACTScan(entry, d=d)
	with bench.show("pmat"):
		pmap = pmat.PmatMap(scan, area, sys=sys)
		pcut = pmat.PmatCut(scan)
		rhs  = enmap.zeros((ncomp,)+shape, area.wcs, dtype)
		div  = enmap.zeros((ncomp,ncomp)+shape, area.wcs, dtype)
		junk = np.zeros(pcut.njunk, dtype)
	# Generate planet cut
	with bench.show("planet cut"):
		planet_cut = cuts.avoidance_cut(d.boresight, d.point_offset, d.site,
				args.planet, R)
	if args.sim:
		if args.noiseless: tod_orig = tod.copy()
		with bench.show("inject"):
			pmap.forward(tod, area)
	# Compute atmospheric model
	with bench.show("atm model"):
		model  = smooth(gapfill.gapfill_joneig(tod,	 planet_cut, inplace=False), d.srate)
	if args.sim and args.noiseless:
		model -= smooth(gapfill.gapfill_joneig(tod_orig, planet_cut, inplace=False), d.srate)
		tod   -= tod_orig
		del tod_orig
	with bench.show("atm subtract"):
		tod -= model
		del model
		tod  = tod.astype(dtype, copy=False)
	# Should now be reasonably clean of correlated noise.
	# Proceed to make simple binned map
	with bench.show("rhs"):
		tod *= ivar[:,None]
		pcut.backward(tod, junk)
		pmap.backward(tod, rhs)
	with bench.show("hits"):
		for i in range(ncomp):
			div[i,i] = 1
			pmap.forward(tod, div[i])
			tod *= ivar[:,None]
			pcut.backward(tod, junk)
			div[i] = 0
			pmap.backward(tod, div[i])
	with bench.show("map"):
		if args.no_pol:
			div = div[0:1,0:1]
			rhs = rhs[0:1]			      
		idiv = array_ops.eigpow(div, -1, axes=[0,1], lim=1e-5, fallback="scalar")
		map  = enmap.map_mul(idiv, rhs)
	if args.no_abscal:
		with bench.show("Undoing abscal"):
			# Undo abscal after solving to avoid numerical instabilities.
			undo_abscal(d, entry, map, div, rhs)
	# Estimate central amplitude
	c = np.array(map.shape[-2:])//2
	crad  = 50
	mcent = map[:,c[0]-crad:c[0]+crad,c[1]-crad:c[1]+crad]
	mcent = enmap.downgrade(mcent, 4)
	amp   = np.max(mcent)
	print("%s amp %7.3f asens %7.3f" % (id, amp/1e6, asens))
	with bench.show("write"):
		np.savetxt("%s%s_stats.txt" % (prefix, bid), np.asarray([[amp, asens]]),
			   header='amp, asens')
		enmap.write_map("%s%s_map.fits" % (prefix, bid), map)
		if not args.no_rhs:
			enmap.write_map("%s%s_rhs.fits" % (prefix, bid), rhs)
		if not args.no_div:
			enmap.write_map("%s%s_div.fits" % (prefix, bid), div)
	del d, scan, pmap, pcut, tod, map, rhs, div, idiv, junk
