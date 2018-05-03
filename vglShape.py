from __future__ import print_function
import numpy as np
import sys

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
				bits per sample
	"""
	def __init__(self, ndim, nChannels, width, height, depht=0):
		ndim = ndim
		if(ndim == 1):
			ndim = 2
			shape = [1, width, height]
		elif(ndim == 2):
			shape = [nChannels, width, height]
		elif(ndim == 3):
			shape = [nChannels, width, height, depht]

		vglCreateShape(shape, ndim)

	def _vglCreateShape(self, shape, ndim, bps=8):

		self.ndim = ndim
		self.bps = bps
		self.size = 1

		if( (bps == 1) and (shape[0] != 1) ):
			print("Imagem com 1 bps e mais de um canal!")
			return 1

		maxi = ndim
		c = shape[0]
		w = shape[1]

		if(ndim == 1):
			maxi == 1

		for i in range(0, 10): # 10 IS THE VGL_MAX_DIM IN THE C++ CODE
			if(i <= maxi):
				self.shape[i] = shape[i]
				
				if(i == 0):
					self.offset[i] = 1
				elif(i == 0):
					self.offset[i] = findWidthStep(bps, w, c)
				else:
					self.offset[i] = shape[i-1] * self.offset[i-1]
			else:
				self.shape[i] = 1
				self.offset[i] = 0
		self.size *= self.shape[maxi] * self.offset[maxi]


	def __tostring__(self):
		return "Sem retorno personalizado configurado"