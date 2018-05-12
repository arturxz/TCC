import numpy as np 
import vglConst as vc
from vglShape import VglShape

class VglClStrEl(object):
	def __init__(self):
		self.data = np.zeros((vc.VGL_ARR_CLSTREL_SIZE()), np.int32)
		self.ndim = 0
		self.shape = np.zeros((vc.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.offset = np.zeros((vc.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.int = 0

class VglStrEl(object):
	def __init__(self):
		self.data = np.zeros((vc.VGL_ARR_CLSTREL_SIZE()), np.int32)
		self.ndim = 0
		self.shape = np.zeros((vc.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.offset = np.zeros((vc.VGL_ARR_SHAPE_SIZE()), np.int32)
		self.int = 0

	def VglCreateStrEl(self, data, vglShape):
		size = vglShape.getSize()
		self.vglShape = VglShape()
		self.vglShape.constructorFromVglShape(vglShape)
		self.data = np.zeros((size), np.float32)

		for i in range(0, size):
			self.data[i] = data[i]

	def constructorFromDataVglShape(self, data, vglShape):
		self.VglCreateStrEl(data, vglShape)

	"""
		In the .cl file, is used the name "type"
		but "type" is a reserved name in Python.
		Then, on this object, the name "Type" will replace "type".
	"""
	def constructorFromTypeNdim(self, Type, ndim):
		shape = np.zeros((vc.VGL_MAX_DIM()), np.int32)
		shape[0] = 1
		shape[2] = 1

		for i in range(1, ndim+1):
			shape[i] = 3

		vglShape = VglShape()
		vglShape.constructorFromShapeNdimBps(shape, ndim)

		size = vglShape.getSize()
		data = np.zeros((size), np.float32)
		index = 0

		if( Type == vc.VGL_STREL_CROSS() ):
			coord = np.zeros((vc.VGL_ARR_SHAPE_SIZE()), np.int32)

			for i in range(0, size):
				data[i] = 0.0

			for d in range(1, ndim+1):
				coord[d] = 1

			index = vglShape.getIndexFromCoord(coord)
			data[index] = 1.0

			for d in range(1, ndim+1):
				coord[d] = 0
				index = vglShape.getIndexFromCoord(coord)
				data[index] = 1.0

				coord[d] = 1

		elif( Type == vc.VGL_STREL_GAUSS() ):
			coord = np.zeros((vc.VGL_ARR_SHAPE_SIZE), np.int32)
			coord[0] = 0
			size = vglShape.getSize()

			for i in range(0, size):
				val = 1.0
				vglShape.getCoordFromIndex(i, coord)

				for d in range(1, ndim+1):
					if( coord[d] == 1 ):
						val *= 0.5
					else:
						val *= 0.25

				data[i] = val


		elif( Type == vc.VGL_STREL_MEAN() ):

			for i in range(0, size):
				data[i] = 1.0 / size

		else( Type == vc.VGL_STREL_CUBE() ):

			for i in range(0, size):
				data[i] = 1.0

		self.constructorFromDataVglShape(data, vglShape)
		del vglShape
		del data

	def getData()
		return self.data

	def getSize()
		return self.vglShape.getSize()

	def getNpixels():
		return self.vglShape.getNpixels()

	def getNdim():
		return self.vglShape.getNdim()

	def getShape():
		return self.vglShape.getShape()

	def getOffset():
		return self.vglShape.getOffset()

	def asVglClStrEl():
		result = VglClStrEl()
		shape = self.vglShape.asVglClShape()
		size = self.getSize()
		
		if( size > vc.VGL_ARR_CLSTREL_SIZE() ):
			print("Error: structuring element size > VGL_ARR_CLSTREL_SIZE. Change this value in vglClStrEl.h to a greater one.")

		result.ndim = self.vglShape.getNdim()
		result.size = self.vglShape.getSize()
		
		for i in range(0, vc.VGL_MAX_DIM()+1):
			result.shape[i] = shape.shape[i]
			result.offset[i] = shape.offset[i]
		
		for i in range(0, size):
			result.data[i] = self.data[i]

		return result






