from skimage import io
import pyopencl as cl
import numpy as np
import sys

# VISIONGL IMPORTS
import vgl_lib as vl

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
"""
	PYTHON'S VGLIMAGE IS SLIGHTLY DIFFERENT  FROM ITS EQUIVALENT.
	ndim DEFAULTS TO 2D DIMENTION IMAGE
	
	img_mode IS HOW THE IMAGE WILL BE TREATED. IF WILL USE THE
			 ND-ARRAY IMAGE AND PASS THE IMAGE DATA TO THE DEVICE
			 AS A BUFFER OF DATA OR IF WILL CREATE A OPENCL IMAGE
			 OBJECT TO PASS ALL THE NEEDED DATA. IT DEFAULTS TO
			 THE SECOND ONE (OPENCL IMAGE OBJECT).
	
	ALL THE CONSTANTS CAN BE FOUND IN vglConst.py file.
"""
class VglImage(object):
	def __init__(self, imgPath, ndim=None, img_mode=vl.IMAGE_CL_OBJECT()):
		# IF THE IMAGE TYPE IS NOT SPECIFIED, A 2D IMAGE WILL BE ASSUMED
		# INICIALIZING DATA
		self.inContext = 0
		self.filename = imgPath
		self.ndim = ndim
		self.depht = 0
		self.nChannels = 0
		self.hasmipmap = 0

		# PYTHON-EXCLUSIVE DATA
		self.img_ram = None
		self.img_device = None
		self.img_sync = False
		self.img_manipulation_mode = None
		if( (img_mode is vl.IMAGE_CL_OBJECT()) or (img_mode is vl.IMAGE_ND_ARRAY()) ):
			self.img_manipulation_mode = img_mode
		else:
			print("ERROR! UNEXISTENT IMAGE MODE!")

		self.last_changed_host = False
		self.last_changed_device = False

		if(self.ndim is None):
			self.ndim = vl.VGL_IMAGE_2D_IMAGE()
			print("Creating 2D Image!")
		elif(self.ndim is vl.VGL_IMAGE_3D_IMAGE()):
			print("Creating 3D Image!")

		# OPENING IMAGE
		self.vglLoadImage(imgPath)
	
	"""
		THIS METHOD INITIATES THE vglShape OBJECT
		TO THE IMAGE. IT LOOKS IF IS A 2D OR A 3D IMAGE
		AND CONSTRUCT THE vglShape OBJECT.

		IN EVERY CHANGE IN THE IMAGE, THIS METHOD MUST BE
		CALLED, TO ASSURES THAT THE vglShape IS SYNCED WITH
		THE ACTUAL IMAGE.
	"""
	def create_vglShape(self):
		if(self.img_ram is not None):
			print("The image was founded. Creating vglShape.")

			self.vglshape = vl.VglShape()
			if( self.ndim == vl.VGL_IMAGE_2D_IMAGE() ):
				print("2D Image")
				if( len(self.img_ram.shape) == 2 ):
					# SHADES OF GRAY IMAGE
					print("VglImage LUMINANCE")
					self.vglshape.constructor2DShape(1, self.img_ram.shape[1], self.img_ram.shape[0])
				elif(len(self.img_ram.shape) == 3):
					# MORE THAN ONE COLOR CHANNEL
					print("VglImage RGB")
					self.vglshape.constructor2DShape(self.img_ram.shape[2], self.img_ram.shape[1], self.img_ram.shape[0])
			elif( self.ndim == vl.VGL_IMAGE_3D_IMAGE() ):
				print("3D Image")
				if( len(self.img_ram.shape) == 3 ):
					# SHADES OF GRAY IMAGE
					print("VglImage LUMINANCE")
					self.vglshape.constructor3DShape( 1, self.img_ram.shape[2], self.img_ram.shape[1], self.img_ram.shape[0] )
				elif(len(self.img_ram.shape) == 4):
					# MORE THAN ONE COLOR CHANNEL
					print("VglImage RGB")
					self.vglshape.constructor3DShape( self.img_ram.shape[3], self.img_ram.shape[2], self.img_ram.shape[1], self.img_ram.shape[0] )
		
			self.img_sync = False
			self.last_changed_host = True
			self.last_changed_device = False
		else:
			print("Impossible to create a vglImage object. host_image is None.")

	"""
		EQUIVALENT TO:
			vglImage.vglLoadImage
			vglImage.vglLoad3dImage
			vglImage.vglLoadNdImage
			vglImage.vglLoadPgm

		THIS METHOD READS THE IMAGE PATH AND OPEN
		IT AS A NUMPY NDARRAY. THROWS A ERROR MESSAGE
		IF THE PATH IS INCORRECT. BUILD THE MAKE THE CALL
		TO CONSTRUCT THE vglShape OBJECT. 
	"""
	def vglLoadImage(self, imgPath):
		try:
			self.img_ram = io.imread(imgPath)
		except FileNotFoundError as fnf:
			print("vglCreateImage: Error loading image from file:", imgPath)    
			print(str(fnf))
		except Exception as e:
			print("Unrecognized error:")
			print(str(e))

		self.img_sync = False
		self.last_changed_host = True
		self.last_changed_device = False

		self.create_vglShape()
	
	"""
		EQUIVALENT TO vglImage.vglImage3To4Channels()
	"""
	def vglImage3To4Channels(self):
		self.rgb_to_rgba()

	"""
		EQUIVALENT TO vglImage.vglImage4To3Channels()
	"""
	def vglImage4To3Channels(self):
		self.rgba_to_rgb()
	
	"""
		EQUIVALENT TO vglImage.vglUpload()
		TREATING ACCORDING TO img_manipulation_mode.
	"""
	def vglUpload(self, ctx, queue):
		if( self.img_manipulation_mode is vl.IMAGE_CL_OBJECT() ):
			self.vglClImageUpload(ctx, queue)
		elif( self.img_manipulation_mode is vl.IMAGE_ND_ARRAY() ):
			self.vglClNdImageUpload(ctx, queue)

	"""
		EQUIVALENT TO vglImage.vglDownload()
		TREATING ACCORDING TO img_manipulation_mode.
	"""
	def vglDownload(self, ctx, queue):
		if( self.img_manipulation_mode is vl.IMAGE_CL_OBJECT() ):
			self.vglClImageDownload(ctx, queue)
		elif( self.img_manipulation_mode is vl.IMAGE_ND_ARRAY() ):
			self.vglClNdImageDownload(ctx, queue)

	"""
		EQUIVALENT TO vglImage.vglImageUpload() (OPENCL IMAGE OBJECT)

		IT TAKES THE RAM-SIDE IMAGE, ALLOCATES THE
		DEVICE-SIDE MEMORY, CONSTRUCTS THE IMAGE OBJECT
		AND UPLOADS THE IMAGE OBJECT TO THE DEVICE.
	"""
	def vglClImageUpload(self, ctx, queue):
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
		cl.enqueue_copy(queue, self.img_device, self.img_ram, origin=origin, region=region, is_blocking=True)

		self.img_sync = False
		self.last_changed_host = False
		self.last_changed_device = True

	"""
		TO OPENCL IMAGE OBJECT, IS QUIVALENT TO:
			vglImage.vglImageDownloadFaster()
			vglImage.vglImageDownload()
			vglImage.vglImageDownloadFBO()
			vglImage.vglImageDownloadPPM()
			vglImage.vglImageDownloadPGM()

		THIS METHOD TAKES THE DEVICE-SIDE OPENCL IMAGE AND
		COPY IT BACK TO THE RAM-SIDE. IN THE END, THE METHOD
		CALL THE vglShape CONSTRUCTOR, TO ATUALIZE THE vglShape
		OBJECT.
	"""
	def vglClImageDownload(self, ctx, queue):
		# MAKE IMAGE DOWNLOAD HERE
		print("Downloading Image from device.")

		if( self.getVglShape().getNFrames() == 1 ):
			origin = ( 0, 0, 0 )
			region = ( self.getVglShape().getWidth(), self.getVglShape().getHeight(), 1 )
			totalSize = self.getVglShape().getHeight() * self.getVglShape().getWidth() * self.getVglShape().getNChannels()

			buffer = np.zeros(totalSize, self.img_ram.dtype)
			cl.enqueue_copy(queue, buffer, self.img_device, origin=origin, region=region, is_blocking=True)

			if( self.getVglShape().getNChannels() == 1 ):
				buffer = np.frombuffer( buffer, self.img_ram.dtype ).reshape( self.getVglShape().getHeight(), self.getVglShape().getWidth() )
			elif( (self.getVglShape().getNChannels() == 3) or (self.getVglShape().getNChannels() == 4) ):
				buffer = np.frombuffer( buffer, self.img_ram.dtype ).reshape( self.getVglShape().getHeight(), self.getVglShape().getWidth(), self.getVglShape().getNChannels() )
		elif( self.getVglShape().getNFrames() > 1 ):
			pitch = (0, 0)
			origin = ( 0, 0, 0 )
			region = ( self.getVglShape().getWidth(), self.getVglShape().getHeight(), self.getVglShape().getNFrames() )
			totalSize = self.getVglShape().getHeight() * self.getVglShape().getWidth() * self.getVglShape().getNFrames()

			buffer = np.zeros(totalSize, self.img_ram.dtype)
			cl.enqueue_copy(queue, buffer, self.img_device, origin=origin, region=region, is_blocking=True)


			if( self.getVglShape().getNChannels() == 1 ):
				buffer = np.frombuffer( buffer, self.img_ram.dtype ).reshape( self.getVglShape().getNFrames(), self.getVglShape().getHeight(), self.getVglShape().getWidth() )
			elif( (self.getVglShape().getNChannels() == 3) or (self.getVglShape().getNChannels() == 4) ):
				buffer = np.frombuffer( buffer, self.img_ram.dtype ).reshape( self.getVglShape().getNFrames(), self.getVglShape().getHeight(), self.getVglShape().getWidth(), self.getVglShape().getNChannels() )

		self.img_ram = buffer
		self.create_vglShape()

		self.img_sync = False
		self.last_changed_device = False
		self.last_changed_host = True

	"""
		EQUIVALENT TO vglImage.vglImageUpload() (OPENCL BUFFER TO ND-IMAGE)

		IT TAKES THE RAM-SIDE IMAGE, ALLOCATES THE
		DEVICE-SIDE MEMORY, AND PASSES THE ND-ARRAY
		TO THE DEVICE.
	"""
	def vglClNdImageUpload(self, ctx, queue):
		print("NdArray image Upload")
		
		# CREATING DEVICE POINTER AND COPYING HOST TO DEVICE
		mf = cl.mem_flags
		self.img_device = cl.Buffer(ctx, mf.READ_ONLY, self.get_host_image().nbytes)
		cl.enqueue_copy(queue, self.img_device, self.get_host_image().tobytes(), is_blocking=True)

		self.img_sync = False
		self.last_changed_host = False
		self.last_changed_device = True

	"""
		TO OPENCL BUFFER IMAGE, IS QUIVALENT TO:
			vglImage.vglImageDownloadFaster()
			vglImage.vglImageDownload()
			vglImage.vglImageDownloadFBO()
			vglImage.vglImageDownloadPPM()
			vglImage.vglImageDownloadPGM()

		THIS METHOD TAKES THE DEVICE-SIDE OPENCL BUFFER AND
		COPY IT BACK TO THE RAM-SIDE. THEN, THE METHOD CALLS
		THE vglShape CONSTRUCTOR, TO ATUALIZE THE vglShape
		OBJECT.
	"""
	def vglClNdImageDownload(self, ctx, queue):
		print("NdArray image Download")

		cl.enqueue_copy(queue, self.img_ram, self.img_device)
		self.create_vglShape()

		self.img_sync = False
		self.last_changed_device = False
		self.last_changed_host = True

	def getVglShape(self):
		return self.vglshape
		
	"""
		THIS METHOD MUST BE CALLED EVERY CHANGE IN
		THE IMAGE, TO ASSURED THAT THE RAM-SIDE AND
		DEVICE-SIDE ARE SYNCED.
	"""
	def sync(self, ctx, queue):
		if( not self.img_sync ):
			if( self.last_changed_device ):
				self.vglDownload(ctx, queue)
			elif( self.last_changed_host ):
				self.vglUpload(ctx, queue)
		else:
			print("Already synced")

	"""
		EQUIVALENT TO DIFFERENT IMAGE SAVE
		METHODS IN vglImage.cpp
	"""
	def img_save(self, name):
		print("Saving Picture in Hard Drive")
		io.imsave(name, self.img_ram)

	"""
		EQUIVALENT TO vglImage.3To4Channels()
	"""
	def rgb_to_rgba(self):
		print("[RGB -> RGBA]")
		img_ram_rgba = np.empty((self.vglshape.getHeight(), self.vglshape.getWidth(), 4), self.img_ram.dtype)

		img_ram_rgba[:,:,0] = self.img_ram[:,:,0]
		img_ram_rgba[:,:,1] = self.img_ram[:,:,1]
		img_ram_rgba[:,:,2] = self.img_ram[:,:,2]
		img_ram_rgba[:,:,3] = 255

		self.img_ram = img_ram_rgba
		self.create_vglShape()

	"""
		EQUIVALENT TO vglImage.3To4Channels()
	"""
	def rgba_to_rgb(self):
		print("[RGBA -> RGB]")
		if( (self.img_ram[0,0,:].size < 4) | (self.img_ram[0,0,:].size > 4) ):
			print("IMAGE IS NOT RGBA")
		else:
			img_ram_rgb = np.empty((self.vglshape.getHeight(), self.vglshape.getWidth(), 3), self.img_ram.dtype)
			img_ram_rgb[:,:,0] = self.img_ram[:,:,0]
			img_ram_rgb[:,:,1] = self.img_ram[:,:,1]
			img_ram_rgb[:,:,2] = self.img_ram[:,:,2]

			self.img_ram = img_ram_rgb
			self.create_vglShape()

	def get_similar_device_image_object(self, ctx, queue):

		if(self.ndim == vl.VGL_IMAGE_2D_IMAGE()):
			shape  = ( self.vglshape.getWidth(), self.vglshape.getHeight() )
			mf = cl.mem_flags
			imgFormat = cl.ImageFormat(self.get_toDevice_channel_order(), self.get_toDevice_dtype())
			img_copy = cl.Image(ctx, mf.WRITE_ONLY, imgFormat, shape)
		elif(self.ndim == vl.VGL_IMAGE_3D_IMAGE()):
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
	
	def get_device_image(self):
		return self.img_device
	
	def get_ram_image(self):
		return self.img_ram
	
	def get_toDevice_dtype(self):
		img_device_dtype = None
		if( self.img_ram.dtype == np.uint8 ):
			img_device_dtype = cl.channel_type.UNORM_INT8
			print("8bit Channel Size!")
		elif( self.img_ram.dtype == np.uint16 ):
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
