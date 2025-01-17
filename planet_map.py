from __future__ import division, print_function
import numpy as np, sys, os
from enlib import utils
with utils.nowarn(): import h5py
from enlib import config, pmat, mpi, errors, gapfill, enmap, bench, sampcut, cg
from enlib import fft, array_ops
from enact import filedb, actscan, actdata, cuts, nmat_measure

# NOTE
fft.set_engine('fftw')

config.set("pmat_cut_type",  "full")

parser = config.ArgumentParser(os.environ["HOME"]+"./enkirc")
parser.add_argument("planet", help="E.g, 'Uranus', can also by RA, DEC string in degrees, e.g. '83.63322, 22.01446' for Tau_A.")
parser.add_argument("area")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("tag", nargs="?")
parser.add_argument("-R", "--dist",    type=float, default=0.2)
parser.add_argument("-e", "--equator", action="count", default=0)
parser.add_argument("-z", "--zenith",  action="count", default=0)
parser.add_argument("-c", "--cont",    action="store_true")
parser.add_argument("--sim",	       type=str,   default=None, help="Passing a sel here sets up simulation mode. The simulations will consist of data from the sim sel TODs with the scanning pattern of the real TODs, and with the signal read off from the area map")
parser.add_argument("--sim-point-sigma", default=None, type=float, help="Shift input simulation map by random Gaussian offset with this standard deviation in arcmin")
parser.add_argument("--sim-point-seed", default=None, type=int, help="Random seed for random point offset number generator")
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
parser.add_argument("-m", "--model",   type=str,   default="joneig")
parser.add_argument("--subtract-buddies", action='store_true', help="Subtract the buddy sidelobes")

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

try:
	planet = np.radians(utils.parse_floats(args.planet))
	planet_str = f'{np.degrees(planet[0]):.6f}_{np.degrees(planet[1]):.6f}'
except ValueError:
	planet = args.planet
	planet_str = planet

if args.equ:
	sys = "equ:"+planet_str
else:
	sys = "hor:"+planet_str
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

	if args.sim_point_sigma:
		seedseq = np.random.SeedSequence(args.sim_point_seed)
		child_seeds = seedseq.spawn(len(ids))
		streams = [np.random.default_rng(s) for s in child_seeds]

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

def calc_model_joneig(tod, cut, srate=400):
	return smooth(gapfill.gapfill_joneig(tod, cut, inplace=False), srate)

def calc_model_constrained(tod, cut, srate=400, mask_scale=0.3, lim=3e-4, maxiter=50, verbose=False):
	# First do some simple gapfilling to avoid messing up the noise model
	tod = sampcut.gapfill_linear(cut, tod, inplace=False)
	ft = fft.rfft(tod) * tod.shape[1]**-0.5
	iN = nmat_measure.detvecs_jon(ft, srate)
	del ft
	iV = iN.ivar*mask_scale
	def A(x):
		x   = x.reshape(tod.shape)
		Ax  = iN.apply(x.copy())
		Ax += sampcut.gapfill_const(cut, x*iV[:,None], 0, inplace=True)
		return Ax.reshape(-1)
	b  = sampcut.gapfill_const(cut, tod*iV[:,None], 0, inplace=True).reshape(-1)
	x0 = sampcut.gapfill_linear(cut, tod).reshape(-1)
	solver = cg.CG(A, b, x0)
	while solver.i < maxiter and solver.err > lim:
		solver.step()
		if verbose:
			print("%5d %15.7e" % (solver.i, solver.err))
	res = solver.x.reshape(tod.shape)
	res = smooth(res, srate)
	return res

def shift_map(imap, offset):
	'''
	Shift input map, simulating a pointing error.
	
	Parameters
	----------
	imap : (..., ny, nx) enmap
	    Input map.
	offset : (2,) array
	    Shift in Y direction and X direction in arcmin.

	Returns
	-------
	omap : (..., ny, nx) enmap
	    Copy of input map that has been shifted. Same WCS as input map.
	'''

	offset = np.asarray(offset) / 60
	offset_pix = offset / imap.wcs.wcs.cdelt[::-1]
	omap = enmap.fractional_shift(imap, offset_pix, keepwcs=True, nofft=False)
	return enmap.samewcs(np.ascontiguousarray(omap), imap)

calc_model = {"joneig": calc_model_joneig, "constrained": calc_model_constrained}[args.model]

for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	bid   = id.replace(":","_")
	entry = filedb.data[id]
	if args.tags: entry.tag = args.tags
	oname = "%s%s_map.fits" % (prefix, bid)
	if args.cont and os.path.isfile(oname):
		print("Skipping %s (already done)" % (id))
		continue
	if args.sim_point_sigma:
		rng = streams[ind]
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
		if d.ndet == 0 or d.nsamp < 2: raise errors.DataMissing("No data left")
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
		if args.subtract_buddies:
			pmat_buddy = pmat.PmatMapMultibeam(
				scan, area, scan.buddy_offs,
				scan.buddy_comps, order=0, sys=sys)
		rhs  = enmap.zeros((ncomp,)+shape, area.wcs, dtype)
		div  = enmap.zeros((ncomp,ncomp)+shape, area.wcs, dtype)
		junk = np.zeros(pcut.njunk, dtype)
	# Generate planet cut
	with bench.show("planet cut"):
		planet_cut = cuts.avoidance_cut(d.boresight, d.point_offset, d.site,
				planet, R)
	if args.sim:
		if args.noiseless: tod_orig = tod.copy()
		with bench.show("inject"):
			if args.sim_point_sigma:
				area_sim = shift_map(area, 
				    rng.normal(scale=args.sim_point_sigma, size=2))
			else:
				area_sim = area
			pmap.forward(tod, area_sim)
	# Compute atmospheric model
	with bench.show("atm model"):
		model  = calc_model(tod, planet_cut, d.srate)
	if args.sim and args.noiseless:
		model -= calc_model(tod_orig, planet_cut, d.srate)
		tod   -= tod_orig
		del tod_orig
	with bench.show("atm subtract"):
		tod -= model
		del model
		tod  = tod.astype(dtype, copy=False)
	# Should now be reasonably clean of correlated noise.
	# Proceed to make simple binned map
	with bench.show("rhs"):
		tod_orig = tod.copy()
		tod *= ivar[:,None]
		pcut.backward(tod, junk)
		pmap.backward(tod, rhs)
		#if args.subtract_buddies:
		#	pmat_buddy.backward(tod, rhs, mmul=+1, tmul=-1)
	with bench.show("hits"):
		for i in range(ncomp):
			div[i,i] = 1
			pmap.forward(tod, div[i])
			#if args.subtract_buddies:
			#	pmat_buddy.forward(tod, div[i], tmul=1)
			tod *= ivar[:,None]
			pcut.backward(tod, junk)
			div[i] = 0
			pmap.backward(tod, div[i])
			#if args.subtract_buddies:
			#	pmat_buddy.backward(tod, div[i], tmul=-1)
	with bench.show("map"):
		if args.no_pol:
			div = div[0:1,0:1]
			rhs = rhs[0:1]			      
		idiv = array_ops.eigpow(div, -1, axes=[0,1], lim=1e-5, fallback="scalar")
		map  = enmap.map_mul(idiv, rhs)
	if args.subtract_buddies:
		with bench.show("Buddy subtraction"):
			pmat_buddy.forward(tod_orig, map, tmul=1, mmul=-1)
			tod_orig *= ivar[:,None]
			pcut.backward(tod_orig, junk)
			rhs *= 0
			pmap.backward(tod_orig, rhs)
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
