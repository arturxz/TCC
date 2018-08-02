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
		img_host_rgba = np.empty((self.vglshape.getHeight(), self.vglshape.getWidth(), 4), self.img_host.dtype)

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
			img_host_rgb = np.empty((self.vglshape.getHeight(), self.vglshape.getWidth(), 3), self.img_host.dtype)
			img_host_rgb[:,:,0] = self.img_host[:,:,0]
			img_host_rgb[:,:,1] = self.img_host[:,:,1]
			img_host_rgb[:,:,2] = self.img_host[:,:,2]

			self.img_host = img_host_rgb
			self.create_vglShape()

	def vglImageUpload(self, ctx, queue):
		# IMAGE VARS
		origin = ( 0, 0, 0 )
		region = ( self.vglshape.getHeight(), self.vglshape.getWidth(), 1 )
		shape  = ( self.vglshape.getHeight(), self.vglshape.getWidth() )

		mf = cl.mem_flags
		imgFormat = cl.ImageFormat(self.get_toDevice_channel_order(), self.get_toDevice_dtype())
		self.img_device = cl.Image(ctx, mf.READ_ONLY, imgFormat, shape)

		# COPYING NDARRAY IMAGE TO OPENCL IMAGE OBJECT
		cl.enqueue_copy(queue, self.img_device, self.img_host.tobytes(), origin=origin, region=region, is_blocking=True)

		self.img_sync = False
		self.last_changed_host = False
		self.last_changed_device = True

	def vglImageDownload(self, ctx, queue):
		# MAKE IMAGE DOWNLOAD HERE
		origin = ( 0, 0, 0 )
		region = ( self.vglshape.getHeight(), self.vglshape.getWidth(), 1 )
		shape  = ( self.vglshape.getHeight(), self.vglshape.getWidth() )

		buffer = np.zeros(self.vglshape.getWidth()*self.vglshape.getHeight()*self.vglshape.getNChannels(), self.img_host.dtype)
		cl.enqueue_copy(queue, buffer, self.img_device, origin=origin, region=region, is_blocking=True)

		if( self.vglshape.getNChannels() == 1 ):
			buffer = np.frombuffer( buffer, self.img_host.dtype ).reshape( self.vglshape.getHeight(), self.vglshape.getWidth() )
		elif( (self.vglshape.getNChannels() == 3) or (self.vglshape.getNChannels() == 4) ):
			buffer = np.frombuffer( buffer, self.img_host.dtype ).reshape( self.vglshape.getHeight(), self.vglshape.getWidth(), self.vglshape.getNChannels() )

		self.img_host = buffer

		self.img_sync = False
		self.last_changed_device = False
		self.last_changed_host = True

	def sync(self, ctx, queue):
		if( not self.img_sync ):
			if( self.last_changed_device ):
				self.vglImageDownload(ctx, queue)
			elif(self.last_changed_host ):
				self.vglImageUpload(ctx, queue)
		else:
			print("Already synced")

	def img_save(self, name):
		print("Saving Picture in Hard Drive")
		io.imsave(name, self.img_host)
	
	def get_similar_device_image_object(self, ctx, queue):

		shape  = ( self.vglshape.getHeight(), self.vglshape.getWidth() )
		
		mf = cl.mem_flags
		imgFormat = cl.ImageFormat(self.get_toDevice_channel_order(), self.get_toDevice_dtype())
		return cl.Image(ctx, mf.WRITE_ONLY, imgFormat, shape)
	
	def set_device_image(self, img):
		if( isinstance(img, cl.Image) ):
			self.img_device = img
			
			self.img_sync = False
			self.last_changed_device = True
			self.last_changed_host = False
		else:
			print("Invalid object. cl.Image objects only.")

	
	def getVglShape(self):
		return self.vglshape
	
	def get_device_image(self):
		return self.img_device
	
	def get_host_image(self):
		return self.img_host
	
	def get_toDevice_dtype(self):
		img_device_dtype = None
		if( self.img_host.dtype == np.uint8 ):
			img_device_dtype = cl.channel_type.UNORM_INT8
		elif( self.img_host.dtype == np.uint8 ):
			img_device_dtype = cl.channel_type.UNORM_INT16

		return img_device_dtype
	
	def get_toDevice_channel_order(self):
		img_device_channel_order = None
		if( self.vglshape.getNChannels() == 1 ):
			img_device_channel_order = cl.channel_order.LUMINANCE
		elif( self.vglshape.getNChannels() == 2 ):
			img_device_channel_order = cl.channel_order.RG
		elif( self.vglshape.getNChannels() == 3 ):
			img_device_channel_order = cl.channel_order.RGB
		elif( self.vglshape.getNChannels() == 4 ):
			img_device_channel_order = cl.channel_order.RGBA
		
		return img_device_channel_order


#ctx = cl.create_some_context()
#queue = cl.CommandQueue(ctx)

#img = VglImage("yamamoto.jpg")
#img.rgb_to_rgba()
#print("shape", img.img_host.shape)
#img.vglImageUpload(ctx, queue)
#img.vglImageDownload(ctx, queue)
#img.rgba_to_rgb()
#img.img_save("saida.jpg")