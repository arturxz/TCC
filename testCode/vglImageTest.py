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
		self.img_sync = False

		self.last_changed_host = False
		self.last_changed_device = False

		# OPENING IMAGE
		self.set_image_host(imgPath)

	def create_vglShape(self):
		if(self.img_host is not None):
			print("The image was founded. Loading data.")

			self.vglshape = VglShape()
			self.vglshape.constructor2DShape(self.img_host.shape[2], self.img_host.shape[1], self.img_host.shape[0])

		self.last_changed_host = True
		self.last_changed_device = False

	def set_image_host(self, imgPath):
		try:
			self.img_host = io.imread(imgPath)
		except FileNotFoundError as fnf:
			print("Image wasn't found. ")    
		except Exception as e:
			print("Unrecognized error:")
			print(str(e))

		self.create_vglShape()

	def rgb_to_rgba(self):
		img_host_rgba = np.empty((self.img_host.shape[0], self.img_host.shape[1], 4), self.img_host.dtype)

		img_host_rgba[:,:,0] = self.img_host[:,:,0]
		img_host_rgba[:,:,1] = self.img_host[:,:,1]
		img_host_rgba[:,:,2] = self.img_host[:,:,2]
		img_host_rgba[:,:,3] = 255

		self.img_host = img_host_rgba
		self.create_vglShape()

	def rgba_to_rgb(self):
		if( (self.img_host[0,0,:].size < 4) | (self.img_host[0,0,:].size > 4) ):
			print("IMAGE IS NOT RGBA")
		else:
			img_host_rgb[:,:,0] = self.img_host[:,:,0]
			img_host_rgb[:,:,1] = self.img_host[:,:,1]
			img_host_rgb[:,:,2] = self.img_host[:,:,2]
			self.img_host = img_host_rgb
			self.create_vglShape()

	def vglImageUpload(self, ctx, queue):
		mf = cl.mem_flags

		# IMAGE VARS
		origin = (0, 0, 0)
		region = (self.getVglShape.getHeight(), self.getVglShape.getWidth(), 1)
		shape = ( self.getVglShape.getWidth(), self.getVglShape.getheight(), self.getVglShape.getNChannels )
		pitch = (0, 0)

		img_device_dtype = None
		if( self.img_host.dtype == np.uint8 ):
			img_device_dtype = cl.channel_type.UNORM_INT8
		elif( self.img_host.dtype == np.uint8 ):
			img_device_dtype = cl.channel_type.UNORM_INT16

		img_device_channel_order = None
		if( self.getVglShape.getNChannels() == 1 ):
			img_device_channel_order = cl.channel_order.LUMINANCE
		elif( self.getVglShape.getNChannels() == 2 ):
			img_device_channel_order = cl.channel_order.RG
		elif( self.getVglShape.getNChannels() == 3 ):
			img_device_channel_order = cl.channel_order.RGB
		elif( self.getVglShape.getNChannels() == 4 ):
			img_device_channel_order = cl.channel_order.RGBA


		imgFormat = cl.ImageFormat(img_device_channel_order, img_device_dtype)
		self.img_device = cl.Image(ctx, mf.READ_ONLY, imgFormat, shape)

		# COPYING NDARRAY IMAGE TO OPENCL IMAGE OBJECT
		cl.enqueue_copy(self.queue, self.img_in_cl, self.img.tobytes(), is_blocking=True)

		self.img_sync = False
		self.last_changed_host = False
		self.last_changed_device = True

	def vglImageUpload(self, ctx, queue):
		# MAKE IMAGE DOWNLOAD HERE

	def sync_img(self):
		if( not self.img_sync ):
			if( self.last_changed_device ):
				self.vglImageDownload()
			elif(self.last_changed_host ):
				self.vglImageUpload()
		else:
			print("Already synced")

	def getVglShape(self):
		return self.vglshape



img = VglImage("test.jpg")
print("W", img.vglshape.getWidth())
print("H", img.vglshape.getHeight())
print("C", img.vglshape.getNChannels())

img.rgb_to_rgba()

print("Wa", img.vglshape_rgba.getWidth())
print("Ha", img.vglshape_rgba.getHeight())
print("Ca", img.vglshape_rgba.getNChannels())
