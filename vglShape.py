import numpy as np
import vglConst as vc

class vglClShape(object):
	def __init__(self, ndim=0, size=0):
		self.ndim = ndim
		self.shape = np.zeros((vc.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.offset = np.zeros((vc.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.size = size

class vglShape(object):

	"""
		ndim
			1D Images: is treated as 2D Image
			2D Images: is 2
			3D Images: is 3
		shape
			2D Images:
				[0] Image channels (RGB=3, RGBA=4, GreyScale=1)
				[1] Image width 
				[2] Image height
			3D Images:
				[0] Image Channels
				[1] Image width
				[2] Image height
				[3] Image depht
		bps
			All Images:
				bits per sample. it defaults to 8.
		
		UNIMPLEMENTED:
		print methods ( print(String:msg) and printInfo() )
	"""
	
	def __init__(self, ndim=0, nChannels=0, width=0, height=0, depht=0):

		# CREATING CLASS DATA
		self.ndim = -1
		self.shape = np.zeros((vc.VGL_MAX_DIM()+1), np.int32)
		self.offset = np.zeros((vc.VGL_MAX_DIM()+1), np.int32)
		self.size = -1
		self.bps = -1

		# METHOD DATA, TO CALL _vglCreateShape
		shape = np.zeros((vc.VGL_MAX_DIM()+1), np.int32)
		ndim = ndim
		if(ndim == 1): # 1D SHAPE CONSTRUCTOR
			ndim = 2
			shape[0] = 1
			shape[1] = width
			shape[2] = height
		elif(ndim == 2): # 2D SHAPE CONSTRUCTOR
			shape[0] = nChannels
			shape[1] = width
			shape[2] = height
		elif(ndim == 3): # 3D SHAPE CONSTRUCTOR
			shape[0] = nChannels
			shape[1] = width
			shape[2] = height
			shape[3] = depht

		_vglCreateShape(shape, ndim)


	def _vglCreateShape(self, shape, ndim, bps=8):

		self.ndim = ndim
		self.bps = bps
		self.size = 1

		if( (bps == 1) and (shape[0] != 1) ):
			print("Error: Image with 1 bps and mode then one color channels(!)")
			return 1

		maxi = ndim
		c = shape[vc.VGL_SHAPE_NCHANNELS()]
		w = shape[vc.VGL_SHAPE_WIDTH()]

		if(ndim == 1):
			maxi == 2

		for i in range(0, vc.VGL_MAX_DIM()):
			if(i <= maxi):
				self.shape[i] = shape[i]
				
				if(i == 0):
					self.offset[i] = 1
				elif(i == 2):
					self.offset[i] = findWidthStep(bps, w, c)
				else:
					self.offset[i] = shape[i-1] * self.offset[i-1]
			else:
				self.shape[i] = 1
				self.offset[i] = 0
		self.size *= self.shape[maxi] * self.offset[maxi]

	"""
		brief Get index from coordinate array.
		Get index from coordinate array. Calculates value of index by multiplying coordinate values 
		by respective offset, and summing up the results.
	"""
	def _getIndexFromCoord(self, coord):
		result = 0

		for d in range(0, ndim):
			result += self.offset[d] * coord[d]

		return result
	
	"""
		brief Get coordinate array from index.
		Get coordinate array from index. Calculates value of coordinates by dividing index 
		by respective offset.
	"""
	def _getCoorFromIndex(self, index, coord):
		ndim = getNdim()
		shape = getShape()
		offset = getOffset()
		ires = index
		idim = 0.0

		for d in range(ndim, 0, -1):
			idim = ires / offset[d-1]
			ires = ires - idim * offset[d-1]
			coord[d] = idim

	"""
		DEFAULT GETTERS
	"""
	def getNdim(self):
		return self.ndim

	def getShape(self):
		return self.shape

	def getOffset(self):
		return self.offset

	def getSize(self):
		return self.size

	def getBps(self):
		return self.bps

	def getNpixels(self):
		return self.size / self.shape[vc.VGL_SHAPE_NCHANNELS()]

	def getNChannels(self):
		return self.shape[vc.VGL_SHAPE_NCHANNELS()]

	def getWidth(self):
		if(self.ndim == 1):
			return self.shape[vc.VGL_SHAPE_WIDTH()] * self.shape[vc.VGL_SHAPE_HEIGHT()]
		return self.shape[_VGL_SHAPE_WIDTH()]

	def getHeight(self):
		if(self.ndim == 1):
			return 1
		return self.shape[vc.VGL_SHAPE_HEIGHT()]

	def getLength(self):
		return self.shape[vc.VGL_SHAPE_LENGTH()]

	def getWidthIn(self):
		return self.shape[vc.VGL_SHAPE_WIDTH()]

	def getHeightIn(self):
		return self.shaé[vc.VGL_SHAPE_HEIGHT()]

	def getNFrames(self):
		nframes = 1
		ndim = getNdim()
		for i in range(3, ndim):
			nframes *= self.shape[i]
		return nframes

	def findBitsPerSample(depht):
		return depht & 255

	def _findWidthStep(self, bps, width, nChannels):
		
		if(bps == 1):
			return (width-1) / (8+1)
		if(bps < 8):
			print("Error: bits per sample < 8 and != 1. Image depth may be wrong.")
			exit(1)
		return (bps / 8) * nChannels * width

	def asVglClShape(self):
		
		result = VglClShape()
		result.ndim = self.ndim
		result.size = self.size
		
		for i in range(0, vc.VGL_MAX_DIM()):
			result.shape[i] = self.shape[i]
			result.offset[i] = self.offset[i]

		if(self.ndim == 1):
			result.shape[vc.VGL_SHAPE_WIDTH()] = self.getWidth()
			result.offset[vc.VGL_SHAPE_WIDTH()] = result.shape[vc.VGL_SHAPE_WIDTH()-1] * result.offset[vc.VGL_SHAPE_WIDTH()-1]
			result.shape[vc.VGL_SHAPE_HEIGHT()] = self.getHeight()
			result.offset[vc.VGL_SHAPE_HEIGHT()] = result.shape[vc.VGL_SHAPE_HEIGHT()-1] * result.offset[vc.VGL_SHAPE_HEIGHT()-1]

		return result

"""
	DEFINING CONSTRUCTORS:
		THE WAY TO CONSTRUCT THE STREL MUST BE SENT TO THE PYTHON CONSTRUCTOR.
		THE OTHER ARGUMENTS MUST BE SENT INSIDE THE DICT OBJECT IN PYTHON.
		REFERENCE TO DICTS CAN BE CONSULTED HERE: https://docs.python.org/3/tutorial/datastructures.html

	'construct': responsable to define way to construct the vglStrEl object.
		1 is compatible to C++ constructor: VglStrEl::VglStrEl(float* data, VglShape* vglShape) 
		2 is compatible to C++ constructor: VglStrEl::VglStrEl(int type, int ndim)
	data: array with convolution elements
	vglShape: shape or dimension sizes, associated with array

"""
class vglStrEl(object):
	def __init__(self, **kwargs):
