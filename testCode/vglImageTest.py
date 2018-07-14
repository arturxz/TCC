from skimage import io
import matplotlib.pyplot as mp
import pyopencl as cl
import numpy as np
import sys

# VISIONGL IMPORTS
from vglShape import *
from vglStrEl import *
import vglConst as vc

class VglImage(object):
	def __init__(self, imgPath):
		# INICIALIZING DATA
		self.img_host = None
		self.img_device = None
		self.img_sync = True

		# OPENING IMAGE
		self.set_image_host(imgPath)

	def set_image_host(self, imgPath):
		try:
			self.img_host = io.imread(imgPath)
		except FileNotFoundError as fnf:
			print("Image wasn't found. ")    
		except Exception as e:
			print("Unrecognized error:")
			print(str(e))

		if(self.img_host is not None):
			print("The image was founded. Loading data.")

			self.vglshape = VglShape()
			self.vglshape.constructor2DShape(self.img_host.shape[2], self.img_host.shape[1], self.img_host.shape[0])

img = VglImage("test.jpg")
print(img.vglshape.getWidth())
print(img.vglshape.getHeight())
print(img.vglshape.getNChannels())
