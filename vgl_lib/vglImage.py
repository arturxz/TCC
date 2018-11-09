from skimage import io
import pyopencl as cl
import numpy as np
import sys

# VISIONGL IMPORTS
from vglShape import *
from vglStrEl import *
import vglConst as vc

"""
	img:
		is the input image
	cl_shape:
		3D Images:
			The OpenCL's default is to be (img_width, img_height, img_depht)
		2D Images:
			The The OpenCL's default is to be (img_width, img_height)
	cl_pitch:
		3D Images (needed):
			The OpenCL's default is to be (img_width*bytes_per_pixel, img_height*img_width*bytes_per_pixel)
			and it is assumed when pitch=(0, 0) is given
		2D Images (optional):
			The OpenCL's default is to be (img_width*bytes_per_pixel)
			and it is assumed when pitch=(0) is given
	cl_origin
		Is the origin of the image, where to start copying the image.
		2D images must have 0 in the z-axis
	cl_region
		Is where to end the copying of the image.
		2D images must have 1 in the z-axis 
"""

class VglImage(object):
	def __init__(self, imgPath, imgDim=vc.VGL_IMAGE_2D_IMAGE()):
		# IF THE IMAGE TYPE IS NOT SPECIFIED, A 2D IMAGE WILL BE ASSUMED
		# INICIALIZING DATA
		self.imgDim = imgDim
		self.img_host = None
		self.img_device = None
		self.img_sync = False

		self.last_changed_host = False
		self.last_changed_device = False

		if(self.imgDim == vc.VGL_IMAGE_2D_IMAGE()):
			print("Creating 2D Image!")
		elif(self.imgDim == vc.VGL_IMAGE_3D_IMAGE()):
			print("Creating 3D Image!")

		# OPENING IMAGE
		self.set_image_host(imgPath)

	def create_vglShape(self):
		if(self.img_host is not None):
			print("The image was founded. Creating vglShape.")

			self.vglshape = VglShape()
			if( self.imgDim == vc.VGL_IMAGE_2D_IMAGE() ):
				print("2D Image")
				if( len(self.img_host.shape) == 2 ):
					# SHADES OF GRAY IMAGE
					print("VglImage LUMINANCE")
					self.vglshape.constructor2DShape(1, self.img_host.shape[1], self.img_host.shape[0])
				elif(len(self.img_host.shape) == 3):
					# MORE THAN ONE COLOR CHANNEL
					print("VglImage RGB")
					self.vglshape.constructor2DShape(self.img_host.shape[2], self.img_host.shape[1], self.img_host.shape[0])
			elif( self.imgDim == vc.VGL_IMAGE_3D_IMAGE() ):
				print("3D Image")
				if( len(self.img_host.shape) == 3 ):
					# SHADES OF GRAY IMAGE
					print("VglImage LUMINANCE")
					self.vglshape.constructor3DShape( 1, self.img_host.shape[2], self.img_host.shape[1], self.img_host.shape[0] )
				elif(len(self.img_host.shape) == 4):
					# MORE THAN ONE COLOR CHANNEL
					print("VglImage RGB")
					self.vglshape.constructor3DShape( self.img_host.shape[3], self.img_host.shape[2], self.img_host.shape[1], self.img_host.shape[0] )
		
			self.img_sync = False
			self.last_changed_host = True
			self.last_changed_device = False
		else:
			print("Impossible to create a vglImage object. host_image is None.")

	def set_image_host(self, imgPath):
		try:
			self.img_host = io.imread(imgPath)
		except FileNotFoundError as fnf:
			print("Image wasn't found. ")    
			print(str(fnf))
		except Exception as e:
			print("Unrecognized error:")
			print(str(e))

		self.img_sync = False
		self.last_changed_host = True
		self.last_changed_device = False

		self.create_vglShape()

	def rgb_to_rgba(self):
		print("[RGB -> RGBA]")
		img_host_rgba = np.empty((self.vglshape.getHeight(), self.vglshape.getWidth(), 4), self.img_host.dtype)

		img_host_rgba[:,:,0] = self.img_host[:,:,0]
		img_host_rgba[:,:,1] = self.img_host[:,:,1]
		img_host_rgba[:,:,2] = self.img_host[:,:,2]
		img_host_rgba[:,:,3] = 255

		self.img_host = img_host_rgba
		self.create_vglShape()

	def rgba_to_rgb(self):
		print("[RGBA -> RGB]")
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
		print("Uploading image to device.")
		if( self.getVglShape().getNFrames() == 1 ):
			origin = ( 0, 0, 0 )
			region = ( self.getVglShape().getWidth(), self.getVglShape().getHeight(), 1 )
			shape  = ( self.getVglShape().getWidth(), self.getVglShape().getHeight() )

			mf = cl.mem_flags
			imgFormat = cl.ImageFormat(self.get_toDevice_channel_order(), self.get_toDevice_dtype())
			self.img_device = cl.Image(ctx, mf.READ_ONLY, imgFormat, shape)
		elif( self.getVglShape().getNFrames() > 1 ):
			origin = ( 0, 0, 0 )
			region = ( self.getVglShape().getWidth(), self.getVglShape().getHeight(), self.getVglShape().getNFrames() )
			shape = ( self.getVglShape().getWidth(), self.getVglShape().getHeight(), self.getVglShape().getNFrames() )

			mf = cl.mem_flags
			imgFormat = cl.ImageFormat(self.get_toDevice_channel_order(), self.get_toDevice_dtype())
			self.img_device = cl.Image(ctx, mf.READ_ONLY, imgFormat, shape)
		else:
			print("VglImage NFrames wrong. NFrames returns:", self.getVglShape().getNFrames() )
			return

		# COPYING NDARRAY IMAGE TO OPENCL IMAGE OBJECT
		cl.enqueue_copy(queue, self.img_device, self.img_host, origin=origin, region=region, is_blocking=True)

		self.img_sync = False
		self.last_changed_host = False
		self.last_changed_device = True

	def vglImageDownload(self, ctx, queue):
		# MAKE IMAGE DOWNLOAD HERE
		print("Downloading Image from device.")

		if( self.getVglShape().getNFrames() == 1 ):
			origin = ( 0, 0, 0 )
			region = ( self.getVglShape().getWidth(), self.getVglShape().getHeight(), 1 )
			totalSize = self.getVglShape().getHeight() * self.getVglShape().getWidth() * self.getVglShape().getNChannels()

			buffer = np.zeros(totalSize, self.img_host.dtype)
			cl.enqueue_copy(queue, buffer, self.img_device, origin=origin, region=region, is_blocking=True)

			if( self.getVglShape().getNChannels() == 1 ):
				buffer = np.frombuffer( buffer, self.img_host.dtype ).reshape( self.getVglShape().getHeight(), self.getVglShape().getWidth() )
			elif( (self.getVglShape().getNChannels() == 3) or (self.getVglShape().getNChannels() == 4) ):
				buffer = np.frombuffer( buffer, self.img_host.dtype ).reshape( self.getVglShape().getHeight(), self.getVglShape().getWidth(), self.getVglShape().getNChannels() )
		elif( self.getVglShape().getNFrames() > 1 ):
			pitch = (0, 0)
			origin = ( 0, 0, 0 )
			region = ( self.getVglShape().getWidth(), self.getVglShape().getHeight(), self.getVglShape().getNFrames() )
			totalSize = self.getVglShape().getHeight() * self.getVglShape().getWidth() * self.getVglShape().getNFrames()

			buffer = np.zeros(totalSize, self.img_host.dtype)
			cl.enqueue_copy(queue, buffer, self.img_device, origin=origin, region=region, is_blocking=True)


			if( self.getVglShape().getNChannels() == 1 ):
				buffer = np.frombuffer( buffer, self.img_host.dtype ).reshape( self.getVglShape().getNFrames(), self.getVglShape().getHeight(), self.getVglShape().getWidth() )
			elif( (self.getVglShape().getNChannels() == 3) or (self.getVglShape().getNChannels() == 4) ):
				buffer = np.frombuffer( buffer, self.img_host.dtype ).reshape( self.getVglShape().getNFrames(), self.getVglShape().getHeight(), self.getVglShape().getWidth(), self.getVglShape().getNChannels() )

		self.img_host = buffer
		self.create_vglShape()

		self.img_sync = False
		self.last_changed_device = False
		self.last_changed_host = True

	def vglNdImageUpload(self, ctx, queue):
		print("NdArray image Upload")
		
		# CREATING DEVICE POINTER AND COPYING HOST TO DEVICE
		mf = cl.mem_flags
		self.img_device = cl.Buffer(ctx, mf.READ_ONLY, self.get_host_image().nbytes)
		cl.enqueue_copy(queue, self.img_device, self.get_host_image().tobytes(), is_blocking=True)

		self.img_sync = False
		self.last_changed_host = False
		self.last_changed_device = True
	
	def vglNdImageDownload(self, ctx, queue):
		print("NdArray image Download")

		cl.enqueue_copy(queue, self.img_host, self.img_device)
		self.create_vglShape()

		self.img_sync = False
		self.last_changed_device = False
		self.last_changed_host = True

	def sync(self, ctx, queue):
		if( not self.img_sync ):
			if( self.last_changed_device ):
				self.vglImageDownload(ctx, queue)
			elif( self.last_changed_host ):
				self.vglImageUpload(ctx, queue)
		else:
			print("Already synced")

	def img_save(self, name):
		print("Saving Picture in Hard Drive")
		io.imsave(name, self.img_host)
	
	def get_similar_device_image_object(self, ctx, queue):

		if(self.imgDim == vc.VGL_IMAGE_2D_IMAGE()):
			shape  = ( self.vglshape.getWidth(), self.vglshape.getHeight() )
			mf = cl.mem_flags
			imgFormat = cl.ImageFormat(self.get_toDevice_channel_order(), self.get_toDevice_dtype())
			img_copy = cl.Image(ctx, mf.WRITE_ONLY, imgFormat, shape)
		elif(self.imgDim == vc.VGL_IMAGE_3D_IMAGE()):
			shape  = ( self.vglshape.getWidth(), self.vglshape.getHeight(), self.vglshape.getNFrames() )
			mf = cl.mem_flags
			imgFormat = cl.ImageFormat(self.get_toDevice_channel_order(), self.get_toDevice_dtype())
			img_copy = cl.Image(ctx, mf.WRITE_ONLY, imgFormat, shape)

		#print("--> Orig:", self.get_device_image().width, self.get_device_image().height, self.get_device_image().depth)
		#print("--> Copy:", img_copy.width, img_copy.height, img_copy.depth)

		return img_copy
	
	def set_device_image(self, img):
		if( isinstance(img, cl.Image) or isinstance(img, cl.Buffer) ):
			self.img_device = img
			
			self.img_sync = False
			self.last_changed_device = True
			self.last_changed_host = False
		else:
			print("Invalid object. cl.Image or cl.Buffer objects only.")
	
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
			print("8bit Channel Size!")
		elif( self.img_host.dtype == np.uint16 ):
			img_device_dtype = cl.channel_type.UNORM_INT16
			print("16bit Channel Size!")

		return img_device_dtype
	
	def get_toDevice_channel_order(self):
		img_device_channel_order = None
		if( self.getVglShape().getNChannels() == 1 ):
			img_device_channel_order = cl.channel_order.LUMINANCE
		elif( self.getVglShape().getNChannels() == 2 ):
			img_device_channel_order = cl.channel_order.RG
		elif( self.getVglShape().getNChannels() == 3 ):
			img_device_channel_order = cl.channel_order.RGB
		elif( self.getVglShape().getNChannels() == 4 ):
			img_device_channel_order = cl.channel_order.RGBA
		
		return img_device_channel_order
