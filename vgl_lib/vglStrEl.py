import numpy as np 
import vgl_lib as vl

"""
	EQUIVALENT TO vglClStrEl struct
	LOCATED IN vglClStrEl.h
"""
class VglClStrEl(object):
	def __init__(self, ndim=0, size=0):
		self.data = np.zeros((vl.VGL_ARR_CLSTREL_SIZE()), np.float32)
		self.ndim = np.int32(0)
		self.shape = np.zeros((vl.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.offset = np.zeros((vl.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.size = np.int32(0)
"""
	EQUIVALENT TO VglStrEl OBJECT
	LOCATED IN vglStrEl.cpp
"""
class VglStrEl(object):
	def __init__(self):
		self.vglShape = vl.VglShape()
		self.data = np.zeros((1), np.float32)

	"""
		EQUIVALENT TO vglStrEl.VglCreateStrEl()
		RECEIVES TWO PARAMETERS:
			data: ARRAY WITH CONVOLUTION WINDOWS ELEMENTS.
			vglShape: vglShape OBJECT TO HELP MANIPULATE THE Structuring Element.
	"""
	def VglCreateStrEl(self, data, vglShape):
		size = vglShape.getSize()
		self.vglShape = vl.VglShape()
		self.vglShape.constructorFromVglShape(vglShape)
		self.data = np.zeros((size), np.float32)

		for i in range(0, size):
			self.data[i] = data[i]

	"""
		EQUIVALENT TO vglStrEl.VglCreateStrEl()
		RECEIVES TWO PARAMETERS:
			data: array with convolution window elements.
			vglShape: shape, or dimension sizes, associated with array.
	"""
	def constructorFromDataVglShape(self, data, vglShape):
		self.VglCreateStrEl(data, vglShape)

	"""
		EQUIVALENT TO THE vglStrEl CONSTRUCTOR
		THAT RECEIVES TWO PARAMETERS:
			type: (VGL_SHAPE_CUBE|VGL_SHAPE_CROSS|VGL_SHAPE_GAUSS)
			ndim: number of dimensions.
		
		ATTENTION:
			In the .cpp file, is used the name "type"
			but "type" is a reserved name in Python.
			Then, on this object, the name "Type" will replace "type".
	"""
	def constructorFromTypeNdim(self, Type, ndim):
		shape = np.zeros(vl.VGL_MAX_DIM(), np.int32)
		shape[0] = 1
		shape[2] = 1

		for i in range(1, ndim+1): # ndim+1 cuz in c++ is for i <= ndim
			shape[i] = 3

		vglShape = vl.VglShape()
		vglShape.constructorFromShapeNdimBps(shape, ndim)

		size = vglShape.getSize()
		data = np.zeros((size), np.float32)
		index = 0

		if( Type == vl.VGL_STREL_CROSS() ):
			coord = np.zeros((vl.VGL_ARR_SHAPE_SIZE()), np.int32)

			for i in range(0, size):
				data[i] = np.float32(0.0)

			for d in range(1, ndim+1):
				coord[d] = np.int32(1)

			index = vglShape.getIndexFromCoord(coord)
			data[index] = np.float32(1.0)

			for d in range(1, ndim+1):
				coord[d] = np.int32(0)
				index = vglShape.getIndexFromCoord(coord)
				data[index] = np.float32(1.0)

				coord[d] = np.int32(2)
				index = vglShape.getIndexFromCoord(coord)
				data[index] = np.float32(1.0)

				coord[d] = 1

		elif( Type == vl.VGL_STREL_GAUSS() ):
			coord = np.zeros((vl.VGL_ARR_SHAPE_SIZE()), np.int32)
			coord[0] = np.int32(0)
			size = vglShape.getSize()

			for i in range(0, size):
				val = np.float32(1.0)
				vglShape.getCoordFromIndex(i, coord)

				for d in range(1, ndim+1):
					if( coord[d] == 1 ):
						val = val * np.float32(0.5)
					else:
						val = val * np.float32(0.25)

				data[i] = val

		elif( Type == vl.VGL_STREL_MEAN() ):
			for i in range(0, size):
				data[i] = 1.0 / size

		elif( Type == vl.VGL_STREL_CUBE() ):
			for i in range(0, size):
				data[i] = 1.0
		else:
			for i in range(0, size):
				data[i] = 1.0
			

		self.constructorFromDataVglShape(data, vglShape)

	"""
		DEFAULT GETTERS
	"""
	def getData(self):
		return self.data

	def getSize(self):
		return self.vglShape.getSize()

	def getNpixels(self):
		return self.vglShape.getNpixels()

	def getNdim(self):
		return self.vglShape.getNdim()

	def getShape(self):
		return self.vglShape.getShape()

	def getOffset(self):
		return self.vglShape.getOffset()

	"""
		EQUIVALENT TO THE vglStrEl.asVglStrEl() METHOD
	"""
	def asVglClStrEl(self):
		result = VglClStrEl()
		shape = self.vglShape.asVglClShape()

		size = self.getSize()
		if( size > vl.VGL_ARR_CLSTREL_SIZE() ):
			print("Error: structuring element size > VGL_ARR_CLSTREL_SIZE. Change this value in vglClStrEl.h to a greater one.")

		result.ndim = np.int32(self.vglShape.getNdim())
		result.size = np.int32(self.vglShape.getSize())
		
		for i in range(0, vl.VGL_MAX_DIM()+1):
			result.shape[i] = np.int32(shape.shape[i])
			result.offset[i] = np.int32(shape.offset[i])
		
		for i in range(0, size):
			result.data[i] = np.int32(self.data[i])

		return result