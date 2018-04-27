# TOD-level point source fitting. This works, and seems to be quite a bit better than
# my filtered map-level approach. For example, see this comparison for a high-S/N source
# in 1378883069.1378883122.ar1:
#  1.4179  0.0209 -3.8517  0.0303  9.7507 26.7
#  1.435          -3.864          13.217  38.4
# So that's a 44% higher S/N, corresponding to 2x more data. Not completely comparable, though,
# since the TOD-one didn't marginalize over position. The position also differs by almost a sigma,
# which it shouldn't considering that they share data. And I trust the tod-level one more.
# However, this takes 1-5 s per likelihood evaluation. A robust fit requires ~500 evaluations,
# which would be 8-42 minutes. And that's using 16 cores! That's too slow. So this one is
# useful for comparing with a faster methods for a few reference tods, but not in general.
# Currently N and P take similar time. Can optimize P more with some effort, but P is dominated
# by ffts, and can't improve much.
from __future__ import division, print_function
import numpy as np, time, h5py, astropy.io.fits, os, sys
from scipy import optimize
from enlib import utils, mpi, errors, fft, mapmaking, config
from enlib import pmat, coordinates, enmap, bench, bunch, nmat
from enact import filedb, actdata, actscan, nmat_measure

config.set("downsample", 1, "Amount to downsample tod by")
config.set("gapfill", "joneig", "Gapfiller to use. Can be 'linear' or 'joneig'")
config.default("pmat_interpol_pad", 10.0, "Number of arcminutes to pad the interpolation coordinate system by")
config.default("pmat_interpol_max_size", 1000000, "Maximum mesh size in pointing interpolation. Worst-case time and memory scale at most proportionally with this.")

parser = config.ArgumentParser(os.environ["HOME"] + "./enkirc")
parser.add_argument("mode", help="Mode to use. Can be srcs or planet. This sets up useful defaults for other arguments")
parser.add_argument("srcdb_or_planet")
parser.add_argument("sel")
parser.add_argument("odir")
parser.add_argument("-R", "--radius",    type=float, default=12)
parser.add_argument("-r", "--res",       type=float, default=0.1)
parser.add_argument("-m", "--minimaps",  action="store_true")
parser.add_argument("-c", "--cont",      action="store_true")
parser.add_argument("-s", "--srcs",      type=str,   default=None)
parser.add_argument("-A", "--minamp",    type=float, default=None)
parser.add_argument("-v", "--verbose",   action="count", default=0)
parser.add_argument("-q", "--quiet",     action="count", default=0)
parser.add_argument("-p", "--perdet",    type=int,   default=None)
args = parser.parse_args()

#config.default("pmat_accuracy", 10.0, "Factor by which to lower accuracy requirement in pointing interpolation. 1.0 corresponds to 1e-3 pixels and 0.1 arc minute in polangle")

def getdef(val, default): return val if val is not None else default

def read_srcs(fname, cols=None):
	toks = fname.split(":")
	fname, rest = toks[0], toks[1:]
	if cols is None:
		if len(rest) == 0: cols = [0,1,2]
		else: cols = [int(w) for w in toks[0].split(",")]
	if fname.endswith(".fits"):
		data = astropy.io.fits.open(fname)[1].data
		return np.array([data.ra*utils.degree,data.dec*utils.degree,data.sn])
	else:
		data = np.loadtxt(fname, usecols=cols).T
		data[:2] *= utils.degree
		return data

filedb.init()
ids     = filedb.scans[args.sel]
comm    = mpi.COMM_WORLD
dtype   = np.float32
ndir    = 1
verbose = args.verbose - args.quiet
R       = args.radius*utils.arcmin
res     = args.res*utils.arcmin
utils.mkdir(args.odir)

# Set up our mode-dependent arguments
if args.mode == "srcs":
	perdet = getdef(args.perdet, 0)>0
	minamp = getdef(args.minamp, 500)*1.0
	planet = None
	srcdata= read_srcs(args.srcdb_or_planet)
	src_sys= "cel"
	bounds = filedb.scans.select(ids).data["bounds"]
	bounds = filedb.scans.select(ids).data["bounds"]
	prune_unreliable_srcs = True
elif args.mode == "planet":
	perdet = getdef(args.perdet, 1)>0
	minamp = 0.0
	planet = args.srcdb_or_planet.capitalize()
	# Coordinates are in relative to the planet itself, so it's fiducially at 0,0
	srcdata= np.array([0,0,10e6])[:,None]
	src_sys= "hor:%s/0_0" % planet
	bounds = None
	prune_unreliable_srcs = False
else:
	print("Unknown mode '%s'" % args.mode)
	sys.exit(1)

class PmatTot:
	def __init__(self, data, srcpos, ndir=1, perdet=False):
		# Build source parameter struct for PmatPtsrc
		self.params = np.zeros([srcpos.shape[-1],ndir,8],np.float)
		self.params[:,:,:2] = srcpos[::-1,None,:].T
		self.params[:,:,5:7] = 1
		# Allow per-detector amplitudes
		if perdet:
			self.params = np.tile(self.params[:,:,None,:],(1,1,data.ndet,1))
		scan = actscan.ACTScan(data.entry, d=data)
		self.psrc = pmat.PmatPtsrc(scan, self.params, sys=src_sys)
		self.pcut = pmat.PmatCut(scan)
		# Extract basic offset
		self.off0 = data.point_correction
		self.off  = self.off0*1
		self.el   = np.mean(data.boresight[2,::100])
		self.point_template = data.point_template
	def set_offset(self, off):
		self.off = off*1
		self.psrc.scan.offsets[:,1:] = actdata.offset_to_dazel(self.point_template + off, [0,self.el])
	def forward(self, tod, amps, pmul=1):
		params = self.params.copy()
		params[...,2]   = amps
		junk = np.zeros(self.pcut.njunk,tod.dtype)
		self.psrc.forward(tod, params, pmul=pmul)
		self.pcut.forward(tod, junk)
	def backward(self, tod, amps=None, pmul=1):
		params = self.params.copy()
		junk = np.zeros(self.pcut.njunk,tod.dtype)
		self.pcut.backward(tod, junk)
		self.psrc.backward(tod, params, pmul=pmul)
		if amps is None: amps = params[...,2]
		else: amps[:] = params[...,2]
		return amps

class NmatTot:
	def __init__(self, data, model=None, window=None):
		model  = config.get("noise_model", model)
		window = config.get("tod_window", window)*data.srate
		nmat.apply_window(data.tod, window)
		self.nmat = nmat_measure.NmatBuildDelayed(model, cut=data.cut_noiseest, spikes=data.spikes[:2].T)
		self.nmat = self.nmat.update(data.tod, data.srate)
		nmat.apply_window(data.tod, window, inverse=True)
		self.model, self.window = model, window
		self.ivar = self.nmat.ivar
	def apply(self, tod):
		nmat.apply_window(tod, self.window)
		self.nmat.apply(tod)
		nmat.apply_window(tod, self.window)
		return tod

class PmatThumbs:
	def __init__(self, data, srcpos, res=0.25*utils.arcmin, rad=20*utils.arcmin, perdet=False, detoff=10*utils.arcmin):
		scan = actscan.ACTScan(data.entry, d=data)
		if perdet:
			# Offset each detector's pointing so that we produce a grid of images, one per detector.
			gside  = int(np.ceil(data.ndet**0.5))
			goffs  = np.mgrid[:gside,:gside] - (gside-1)/2.0
			goffs  = goffs.reshape(2,-1).T[:data.ndet]*detoff
			scan.offsets = scan.offsets.copy()
			scan.offsets[:,1:] += goffs
			rad    = rad + np.max(np.abs(goffs))
		# Build geometry for each source
		shape, wcs = enmap.geometry(pos=[[-rad,-rad],[rad,rad]], res=res, proj="car")
		area = enmap.zeros((3,)+shape, wcs, dtype=data.tod.dtype)
		self.pmats = []
		for i, pos in enumerate(srcpos.T):
			if planet: sys = src_sys
			else:      sys = ["icrs",np.array([[pos[0]],[pos[1]],[0],[0]])]
			self.pmats.append(pmat.PmatMap(scan, area, sys=sys))
		self.shape = (len(srcpos.T),3)+shape
		self.wcs   = wcs
	def forward(self, tod, map):
		for i, p in enumerate(self.pmats):
			p.forward(tod, map[i])
	def backward(self, tod, map):
		for i, p in enumerate(self.pmats):
			p.backward(tod, map[i])

class ThumbMapper:
	def __init__(self, data, srcpos, pcut, nmat, perdet=False):
		pthumb = PmatThumbs(data, srcpos, perdet=perdet)
		twork  = np.full(data.tod.shape, 1.0, data.tod.dtype)
		nmat.white(twork)
		div   = enmap.zeros(pthumb.shape, pthumb.wcs, data.tod.dtype)
		junk  = np.zeros(pcut.njunk,data.tod.dtype)
		pcut.backward(twork, junk)
		pthumb.backward(twork, div)
		div = div[:,0]
		self.pthumb, self.pcut, self.nmat = pthumb, pcut, nmat
		self.div = div
	def map(self, tod):
		junk = np.zeros(self.pcut.njunk,tod.dtype)
		rhs  = enmap.zeros(self.pthumb.shape, self.pthumb.wcs, tod.dtype)
		#self.nmat.white(tod)
		self.nmat.apply(tod)
		self.pcut.backward(tod, junk)
		self.pthumb.backward(tod, rhs)
		rhs /= self.div[:,None]
		return rhs

class Likelihood:
	def __init__(self, data, srcpos, srcamp, perdet=False):
		# Set up fiducial source model. These source parameters
		# are not the same as those we will be optimizing.
		self.P = PmatTot(data, srcpos, perdet=perdet)
		self.N = NmatTot(data)
		self.tod  = data.tod # might only need the one below
		self.Nd   = self.N.apply(self.tod.copy())
		self.i    = 0
		# Initial values
		self.amp0   = srcamp[:,None]
		self.off0   = self.P.off0
		self.chisq0 = None
		# These are for internal mapmaking
		self.thumb_mapper = ThumbMapper(data, srcpos, self.P.pcut, self.N.nmat, perdet=perdet)
		self.amp_unit, self.off_unit = 1e3, utils.arcmin
	#def zip(self, off, amps): return np.concatenate([off/self.off_unit, amps[:,0]/self.amp_unit],0)
	#def unzip(self, x): return x[:2]*self.off_unit, x[2:,None]*self.amp_unit
	def zip(self, off): return off/self.off_unit
	def unzip(self, x): return x*self.off_unit
	def fit_amp(self):
		"""Compute the ML amplitude for each point source, along with their covariance.
		This assumes independent source amplitudes. For perdet mapping, this means we may
		need to use detector-diagonal noise."""
		rhs = self.P.backward(self.Nd)
		work = np.zeros(self.tod.shape, self.tod.dtype)
		self.P.forward(work, rhs*0+1)
		self.N.apply(work)
		div  = self.P.backward(work)
		div[div==0] = 1
		return rhs/div, div
	def calc_chisq_fixamp(self, off):
		self.P.set_offset(off)
		amps = self.amp0
		Nr = self.tod.copy()
		self.P.forward(Nr, amps, pmul=-1)
		self.N.apply(Nr)
		PNPa = self.P.backward(Nr)
		return -np.sum(PNPa*amps), amps, amps*0
	def calc_chisq_fitamp(self, off):
		self.P.set_offset(off)
		ahat, aicov = self.fit_amp()
		return -np.sum(ahat**2*aicov), ahat, aicov
	def chisq_wrapper(self, method="fitamp", thumb_path=None, thumb_interval=0, verbose=True):
		if method == "fitamp": fun = self.calc_chisq_fitamp
		else:                  fun = self.calc_chisq_fixamp
		def wrapper(off, full=False):
			t1 = time.time()
			chisq, amps, aicov = fun(self.unzip(off))
			t2 = time.time()
			if thumb_path and thumb_interval and self.i % thumb_interval == 0:
				tod2 = self.tod*0
				self.P.forward(tod2, amps, pmul=1)
				data   = self.thumb_mapper.map(self.tod.copy())
				model  = self.thumb_mapper.map(tod2)
				resid  = data-model
				thumbs = enmap.samewcs([data,model,resid],data)
				enmap.write_map(thumb_path % self.i, thumbs)
				del tod2, thumbs
			if self.chisq0 is None: self.chisq0 = chisq
			if verbose:
				msg = "%4d %6.3f %6.3f" % (self.i,off[0],off[1])
				famps, faicov = amps.reshape(-1), aicov.reshape(-1)
				for i in range(len(famps)):
					nsigma = (famps[i]**2*faicov[i])**0.5
					msg += " %7.3f %4.1f" % (famps[i]/self.amp_unit, nsigma)
				msg += " %12.5e %7.2f" % (self.chisq0-chisq, t2-t1)
				print(msg)
			self.i += 1
			if not full:
				return chisq
			else:
				return chisq, amps, aicov
		return wrapper

# (d-Pa)'N"(d-Pa) = d'N"d + (Pa)'N"(Pa) - 2*d'N"Pa

# If the point sources are far enough away from each other, then they will
# be indepdendent from each other, and all their amplitudes can be fit in
# a single evaluation. Normally you would do:
#  amps = (P'N"P)" P'N"d
# where you need to evaluate P'N"P via unit vector bashing. But if you know
# that it's diagonal, then you can use a single non-unit vector instead:
# diag(P'N"P ones(nsrc)). But should check how good an approximation this is.
# Intuitively, it's a good approximation if the shadow from one souce doesn't
# touch that from another.
#
# P(pos) = int_amp P(pos,amp|d) damp
#        = K int_amp exp(-0.5*(d-Pa)'N"(d-Pa))
#        = K int_amp exp(-0.5*[
#             a'P'N"P(a-(P'N"P)"P'N"d


# (d-Pa)'N"(d-Pa) = d'N"d - 2d'N"Pa + (Pa)'N"Pa

# Load source database
srcpos, amps = srcdata[:2], srcdata[2]
# Which sources pass our requirements?
allowed  = set(range(amps.size))
allowed &= set(np.where(amps > args.minamp)[0])
if args.srcs is not None:
	selected = [int(w) for w in args.srcs.split(",")]
	allowed &= set(selected)

# Iterate over tods
for ind in range(comm.rank, len(ids), comm.size):
	id    = ids[ind]
	oid   = id.replace(":","_")

	# Check if we hit any of the sources. We first make sure
	# there's no angle wraps in the bounds, and then move the sources
	# to the same side of the sky. bounds are pretty approximate, so
	# might not actually hit all these sources
	if bounds is not None:
		poly      = bounds[:,:,ind]*utils.degree
		poly[0]   = utils.rewind(poly[0],poly[0,0])
		# bounds are defined in celestial coordinates. Must convert srcpos for comparison
		mjd       = utils.ctime2mjd(float(id.split(".")[0]))
		srccel    = coordinates.transform(src_sys, "cel", srcpos, time=mjd)
		srccel[0] = utils.rewind(srccel[0], poly[0,0])
		sids      = np.where(utils.point_in_polygon(srccel.T, poly.T))[0]
		sids      = sorted(list(set(sids)&allowed))
	else:
		sids = sorted(list(allowed))
	if len(sids) == 0:
		print("%s has 0 srcs: skipping" % id)
		continue
	try:
		nsrc = len(sids)
		print("%s has %d srcs: %s" % (id,nsrc,", ".join(["%d (%.1f)" % (i,a) for i,a in zip(sids,amps[sids])])))
	except TypeError as e:
		print("Weird: %s" % e)
		print(sids)
		print(amps)
		continue

	# Read the data
	entry = filedb.data[id]
	try:
		data = actdata.read(entry, exclude=["tod"], verbose=verbose)
		#data.restrict(dets=data.dets[:12])
		data+= actdata.read_tod(entry)
		data = actdata.calibrate(data, exclude=["autocut"], verbose=verbose)
		# Avoid planets while building noise model
		if planet is not None:
			data.cut_noiseest *= actdata.cuts.avoidance_cut(data.boresight, data.point_offset, data.site, planet, R)
		if data.ndet < 2 or data.nsamp < 1: raise errors.DataMissing("no data in tod")
	except errors.DataMissing as e:
		print("%s skipped: %s" % (id, e))
		continue
	print("tod shape: [%d,%d]" % data.tod.shape)
	# Prepeare our samples
	#data.tod -= np.mean(data.tod,1)[:,None]
	data.tod -= data.tod[:,None,0].copy()
	data.tod  = data.tod.astype(dtype)
	# Set up our likelihood
	L = Likelihood(data, srcpos[:,sids], amps[sids])
	# Find out which sources are reliable, so we don't waste time on bad ones
	if prune_unreliable_srcs:
		_, aicov = L.fit_amp()
		print(aicov.shape, amps[sids].shape)
		print(amps[sids], aicov**0.5)
		good = amps[sids]**2*aicov[:,0] > 1
		sids = [sid for sid,g in zip(sids,good) if g]
		nsrc = len(sids)
		print("Restricted to %d srcs: %s" % (nsrc,", ".join(["%d (%.1f)" % (i,a) for i,a in zip(sids,amps[sids])])))
	if nsrc == 0: continue
	L = Likelihood(data, srcpos[:,sids], amps[sids], perdet=perdet)
	# And minimize chisq
	x0 = L.zip(L.off0)
	#likfun = L.chisq_wrapper(method="fitamp", thumb_path=args.odir + "/thumb%03d.fits", thumb_interval=1)
	pos    = optimize.fmin_powell(likfun,x0)
	chisq, oamps, oaicov = likfun(pos, full=True)

	# Output our fit
	with h5py.File(args.odir + "/fit_%s.hdf" % oid, "w") as ofile:
		ofile["pos"]  = pos
		ofile["dets"] = data.dets
		ofile["amps"] = oamps
		ofile["aicov"] = oaicov
		ofile["ivar"] = L.N.ivar

	# Our likelihood is
	# (d-Pa)'N"(d-Pa)
	# = d'N"d + a'P'N"Pa - 2 a'P'N"d
	# = chi0  + a'P'N"(Pa - 2 d)
	# However, dot(d-Pa,N"(d-Pa)) is faster than using P't. So no need to factor out chi0

	# But this is linear in a, so we can solve a directly.
	# a <- N(a_ml, acov), where a_ml = (P'N"P)"P'N"Pd and acov = (P'N"P)"
	# If the point sources are independent, then this requires 3 evaluations
	# to build. If they are dependent, then it will be 3*nsrc. If sources
	# are close to the edge, then forwards and backwards going scans
	# will not be independent either, and you get 3*ndir*nsrc.
	# This would need to be repeated every time P changes.
	# On the other hand, plain MC or nonlinear minimization also takes
	# about that long. This would need to 