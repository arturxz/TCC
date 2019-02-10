import numpy as np
import vgl_lib as vl

"""
	EQUIVALENT TO vglClShape, 
	LOCATED IN  vglClShape.h
"""
class VglClShape(object):
	def __init__(self, ndim=0, size=0):
		self.shape = np.zeros((vl.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.offset = np.zeros((vl.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.ndim = np.int32(ndim)
		self.size = np.int32(size)

"""
	EQUIVALENT TO vglShape.cpp
"""
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
				[2] Image heigth
			3D Images:
				[0] Image Channels (RGB=3, RGBA=4, GreyScale=1)
				[1] Image width
				[2] Image heigth
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
		self.shape = np.zeros((vl.VGL_MAX_DIM()+1), np.int32)
		self.offset = np.zeros((vl.VGL_MAX_DIM()+1), np.int32)
		self.size = -1
		self.bps = 8

	"""
		EQUIVALENT TO vglShape.vglCreateShape()

		TAKES SHAPE, NDIM AND BPS (DEFAULT TO 8)
		AND BUILD THE vglShape STRUCTURE.
	"""
	def vglCreateShape(self, shape, ndim, bps=8):

		self.ndim = ndim
		self.bps = bps
		self.size = 1

		if( (bps == 1) and (shape[0] != 1) ):
			print("vglShape: vglCreateShape Error: Image with 1 bps and mode then one color channels(!)")
			exit()

		maxi = ndim
		c = shape[vl.VGL_SHAPE_NCHANNELS()]
		w = shape[vl.VGL_SHAPE_WIDTH()]

		if(ndim == 1):
			maxi == 2

		for i in range(0, vl.VGL_MAX_DIM()+1):
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

	"""
		EQUIVALENT TO THE GENERIC vglShape CONSTRUCTOR
		THAT RECEIVES AN vglShape AS ARGUMENT AND CONSTRUCTS
		A SHAPE FROM ANOTHER SHAPE.
	"""
	def constructorFromVglShape(self, vglShape):
		self.vglCreateShape(vglShape.getShape(), vglShape.getNdim(), vglShape.getBps())

	"""
		EQUIVALENT TO A GENERIC vglShape CONSTRUCTOR
		THAT RECEIVES THREE PARAMETERS:
			shape: ARRAY WITH DIMENSIONS SIZES, WITH THE CHANNEL NUMBERS IN POSITION 0.
    		ndim: NUMBER OF DIMENSIONS
    		bps: BITS PER SAMPLE. DEFAULTS TO 8.
	"""
	def constructorFromShapeNdimBps(self, shape, ndim, bps=8):
		self.vglCreateShape(shape, ndim, bps)

	"""
		EQUIVALENT TO THE 1D vglShape CONSTRUCTOR.
		RECEIVES TWO PARAMETERS:
		w: width
		h: heigth
	"""
	def constructor1DShape(self, w, h):
		shape = np.ones((vl.VGL_MAX_DIM()+1), np.int32)
		ndim = 2
		shape[0] = 1
		shape[1] = w
		shape[2] = h

		self.vglCreateShape(shape, ndim)

	"""
		EQUIVALENT TO THE 2D vglShape CONSTRUCTOR.
		RECEIVES THREE PARAMETERS:
		nChannels: number of channels
		w: width
		h: heigth
	"""
	def constructor2DShape(self, nChannels, w, h):
		shape = np.ones((vl.VGL_MAX_DIM()+1), np.int32)
		ndim = 2
		shape[0] = nChannels
		shape[1] = w
		shape[2] = h

		self.vglCreateShape(shape, ndim)

	"""
		EQUIVALENT TO THE 3D vglShape CONSTRUCTOR.
		RECEIVES FOUR PARAMETERS:
		nChannels: number of channels
		w: width
		h: heigth
		d3: numer of frames
	"""
	def constructor3DShape(self, nChannels, w, h, d3):
		shape = np.ones((vl.VGL_MAX_DIM()+1), np.int32)
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
		EQUIVALENT TO vglShape GETTERS
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
		return self.size / self.shape[vl.VGL_SHAPE_NCHANNELS()]

	def getNChannels(self):
		return self.shape[vl.VGL_SHAPE_NCHANNELS()]

	def getWidth(self):
		if(self.ndim == 1):
			return self.shape[vl.VGL_SHAPE_WIDTH()] * self.shape[vl.VGL_SHAPE_HEIGTH()]
		return self.shape[vl.VGL_SHAPE_WIDTH()]

	def getHeigth(self):
		if(self.ndim == 1):
			return 1
		return self.shape[vl.VGL_SHAPE_HEIGTH()]

	def getLength(self):
		return self.shape[vl.VGL_SHAPE_LENGTH()]

	def getWidthIn(self):
		return self.shape[vl.VGL_SHAPE_WIDTH()]

	def getHeigthIn(self):
		return self.shape[vl.VGL_SHAPE_HEIGTH()]

	def getNFrames(self):
		nframes = 1
		ndim = self.getNdim()
		for i in range(3, ndim+1):
			nframes *= self.shape[i]
		return nframes

	"""
		EQUIVALENT TO THE vglShape.findBitsPerSample(int depht)
	"""
	def findBitsPerSample(self, depht):
		return depht & 255

	"""
		EQUIVALENT TO vglShape.findWidthStep(int bps, int width, int nChannels)
	"""
	def findWidthStep(self, bps, width, nChannels):
		
		if(bps == 1):
			return (width-1) / (8+1)
		if(bps < 8):
			print("Error: bits per sample < 8 and != 1. Image depth may be wrong.")
			exit(1)
		return (bps / 8) * nChannels * width

	"""
		EQUIVALENT TO vglShape.asVglClShape()
	"""
	def asVglClShape(self):
		
		result = VglClShape()
		result.ndim = np.int32(self.ndim)
		result.size = np.int32(self.size)
		
		for i in range(0, vl.VGL_MAX_DIM()+1):
			result.shape[i] = np.int32(self.shape[i])
			result.offset[i] = np.int32(self.offset[i])

		if(self.ndim == 1):
			result.shape[vl.VGL_SHAPE_WIDTH()] = np.int32(self.getWidth())
			result.offset[vl.VGL_SHAPE_WIDTH()] = np.int32(result.shape[vl.VGL_SHAPE_WIDTH()-1] * result.offset[vl.VGL_SHAPE_WIDTH()-1])
			result.shape[vl.VGL_SHAPE_HEIGTH()] = np.int32(self.getHeigth())
			result.offset[vl.VGL_SHAPE_HEIGTH()] = np.int32(result.shape[vl.VGL_SHAPE_HEIGTH()-1] * result.offset[vl.VGL_SHAPE_HEIGTH()-1])

		return result

	"""
		ON C/C++ VERSION, asVglClShape WOULD BE ENOUGH.

		PYTHON-SIDE MUST TREAT THIS DATA BEFORE RETURN IT.
		HERE FOLLOWS THOSE TREATMENTS 
	"""
	def get_asVglClShape_buffer(self):
		result = self.asVglClShape()
		struct_sizes = vl.get_struct_sizes()
		shape_obj = np.zeros(struct_sizes[6], np.uint8)

		self.copy_into_byte_array(result.ndim,	shape_obj, struct_sizes[7])
		self.copy_into_byte_array(result.shape,	shape_obj, struct_sizes[8])
		self.copy_into_byte_array(result.offset,shape_obj, struct_sizes[9])
		self.copy_into_byte_array(result.size,	shape_obj, struct_sizes[10])

		return vl.get_vglshape_opencl_buffer(shape_obj)

	"""
		PYTHON-ONLY METHODS
	"""
	def copy_into_byte_array(self, value, byte_array, offset):
		for iterator, byte in enumerate( value.tobytes() ):
			byte_array[iterator+offset] = byte