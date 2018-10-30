import numpy as np
import vglConst as vc

class VglClShape(object):
	def __init__(self, ndim=0, size=0):
		self.shape = np.zeros((vc.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.offset = np.zeros((vc.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.ndim = np.int32(ndim)
		self.size = np.int32(size)


class VglShape(object):

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
				[0] Image Channels (RGB=3, RGBA=4, GreyScale=1)
				[1] Image width
				[2] Image height
				[3] Image depht
		bps
			All Images:
				bits per sample. it defaults to 8.
		
		UNIMPLEMENTED:
		print methods ( print(String:msg) and printInfo() )
	"""
	
	def __init__(self):

		# CREATING CLASS DATA
		self.ndim = -1
		self.shape = np.zeros((vc.VGL_MAX_DIM()+1), np.int32)
		self.offset = np.zeros((vc.VGL_MAX_DIM()+1), np.int32)
		self.size = -1
		self.bps = 8

	def vglCreateShape(self, shape, ndim, bps=8):

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

		for i in range(0, vc.VGL_MAX_DIM()+1):
			if(i <= maxi):
				self.shape[i] = shape[i]
				
				if(i == 0):
					self.offset[i] = 1
				elif(i == 2):
					self.offset[i] = self.findWidthStep(bps, w, c)
				else:
					self.offset[i] = shape[i-1] * self.offset[i-1]
			else:
				self.shape[i] = 1
				self.offset[i] = 0
		self.size *= self.shape[maxi] * self.offset[maxi]

	def constructorFromVglShape(self, vglShape):
		self.vglCreateShape(vglShape.getShape(), vglShape.getNdim(), vglShape.getBps())

	def constructorFromShapeNdimBps(self, shape, ndim, bps=8):
		self.vglCreateShape(shape, ndim, bps)

	def constructor1DShape(self, w, h):
		shape = np.ones((vc.VGL_MAX_DIM()+1), np.int32)
		ndim = 2
		shape[0] = 1
		shape[1] = w
		shape[2] = h

		self.vglCreateShape(shape, ndim)

	def constructor2DShape(self, nChannels, w, h):
		shape = np.ones((vc.VGL_MAX_DIM()+1), np.int32)
		ndim = 2
		shape[0] = nChannels
		shape[1] = w
		shape[2] = h

		self.vglCreateShape(shape, ndim)

	def constructor3DShape(self, nChannels, w, h, d3):
		shape = np.ones((vc.VGL_MAX_DIM()+1), np.int32)
		ndim = 3
		shape[0] = nChannels
		shape[1] = w
		shape[2] = h
		shape[3] = d3

		self.vglCreateShape(shape, ndim)

	"""
		brief Get index from coordinate array.
		Get index from coordinate array. Calculates value of index by multiplying coordinate values 
		by respective offset, and summing up the results.
	"""
	def getIndexFromCoord(self, coord):
		result = 0

		for d in range(0, self.getNdim()+1):
			result += self.offset[d] * coord[d]

		return result
	
	"""
		brief Get coordinate array from index.
		Get coordinate array from index. Calculates value of coordinates by dividing index 
		by respective offset.
	"""
	def getCoordFromIndex(self, index, coord):
		ndim = self.getNdim()
		shape = self.getShape()
		offset = self.getOffset()
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
		return self.shape[vc.VGL_SHAPE_WIDTH()]

	def getHeight(self):
		if(self.ndim == 1):
			return 1
		return self.shape[vc.VGL_SHAPE_HEIGHT()]

	def getLength(self):
		return self.shape[vc.VGL_SHAPE_LENGTH()]

	def getWidthIn(self):
		return self.shape[vc.VGL_SHAPE_WIDTH()]

	def getHeightIn(self):
		return self.shape[vc.VGL_SHAPE_HEIGHT()]

	def getNFrames(self):
		nframes = 1
		ndim = self.getNdim()
		for i in range(3, ndim+1):
			nframes *= self.shape[i]
		return nframes

	def findBitsPerSample(depht):
		return depht & 255

	def findWidthStep(self, bps, width, nChannels):
		
		if(bps == 1):
			return (width-1) / (8+1)
		if(bps < 8):
			print("Error: bits per sample < 8 and != 1. Image depth may be wrong.")
			exit(1)
		return (bps / 8) * nChannels * width

	def asVglClShape(self):
		
		result = VglClShape()
		result.ndim = np.int32(self.ndim)
		result.size = np.int32(self.size)
		
		for i in range(0, vc.VGL_MAX_DIM()+1):
			result.shape[i] = np.int32(self.shape[i])
			result.offset[i] = np.int32(self.offset[i])

		if(self.ndim == 1):
			result.shape[vc.VGL_SHAPE_WIDTH()] = np.int32(self.getWidth())
			result.offset[vc.VGL_SHAPE_WIDTH()] = np.int32(result.shape[vc.VGL_SHAPE_WIDTH()-1] * result.offset[vc.VGL_SHAPE_WIDTH()-1])
			result.shape[vc.VGL_SHAPE_HEIGHT()] = np.int32(self.getHeight())
			result.offset[vc.VGL_SHAPE_HEIGHT()] = np.int32(result.shape[vc.VGL_SHAPE_HEIGHT()-1] * result.offset[vc.VGL_SHAPE_HEIGHT()-1])

		return result


