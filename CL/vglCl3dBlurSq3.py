from skimage import io
import matplotlib.pyplot as mp
import pyopencl as cl
import numpy as np
import sys

"""
	img:
		is the input image
	img_shape:
		3D Images:
			The OpenCL's default is to be (img_width, img_height, img_depht)
		2D Images:
			The The OpenCL's default is to be (img_width, img_height)
	img_pitch:
		3D Images (needed):
			The OpenCL's default is to be (img_width*bytes_per_pixel, img_height*img_width*bytes_per_pixel)
			and it is assumed if pitch=(0, 0) is given
		2D Images (optional):
			The OpenCL's default is to be (img_width*bytes_per_pixel)
			and it is assumed if pitch=(0) is given
	img_origin
		Is the origin of the image, where to start copying the image.
		2D images must have 0 in the z-axis
	img_region
		Is where to end the copying of the image.
		2D images must have 1 in the z-axis 
"""

class vgl:
	# THE vgl CONSTRUCTOR CREATES A NEW CONTEXT
	# AND INITIATES THE QUEUE, ADDING QUE CONTEXT TO IT.
	def __init__(self):
		print("Starting OpenCL")
		self.platform = cl.get_platforms()[0]
		self.devs = self.platform.get_devices()
		self.device = self.devs[0]
		self.ctx = cl.Context([self.device])
		#self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx)
		self.builded = False

	# THIS FUNCTION WILL LOAD THE KERNEL FILE
	# AND BUILD IT IF NECESSARY.
	def loadCL(self, filepath):
		print("Loading OpenCL Kernel")
		self.kernel_file = open(filepath, "r")

		if ((self.builded == False)):
			self.pgr = cl.Program(self.ctx, self.kernel_file.read())
			self.pgr.build()
			self.kernel_file.close()
			self.builded = True
		else:
			print("Kernel already builded. Going to next step...")

	def loadImage(self, imgpath):
		print("Opening image to be processed")
		self.mf = cl.mem_flags

		# GETTING INPUT IMAGE AND SETTING IT'S PROPERTIES THE IMAGE IS OPENED AS SHAPE (DEPHT, HEIGHT, WIDTH).
		self.img = io.imread(imgpath, plugin='tifffile')

		# GETTING DATA ABOUT THE IMAGE
		self.img_dtype = self.img.dtype
		self.img_ndim = self.img.ndim
		self.img_shape = (self.img.shape[2], self.img.shape[1], self.img.shape[0])
		self.img_origin = (0, 0, 0)
		self.img_region = (self.img_shape[0], self.img_shape[1], self.img_shape[2])
		self.img_pitch = (0, 0)

		# GETTING CL CHANNEL TYPE 
		if( self.img.dtype == np.uint8 ):
			self.img_dtype_cl = cl.channel_type.UNORM_INT8
		elif( self.img.dtype == np.uint16 ):
			self.img_dtype_cl = cl.channel_type.UNORM_INT16
		elif( self.img.dtype == np.int32 ):
			self.img_dtype_cl = cl.channel_type.SIGNED_INT32

		if( self.img_ndim == 3 ):
			# THEN IS A SHADES OF GRAY IMAGE
			self.img_channel_order_cl = cl.channel_order.LUMINANCE
			self.img_nchannels = 1
		elif( self.img_ndim == 4 ):
			# THEN IS A 3D-IMAGE WITH MORE THEN ONE COLOR CHANNEL
			self.img_channel_order_cl = cl.channel_order.RGBA

			if( self.img[0,0,0,:].size == 3 ):
				# THEN IT IS RGB
				self.img_nchannels = 3

				self.img_aux = np.empty((self.img_shape[2],self.img_shape[1], self.img_shape[0], 4), self.img_dtype)
				self.img_aux[:,:,:,0] = self.img[:,:,:,0]
				self.img_aux[:,:,:,1] = self.img[:,:,:,1]
				self.img_aux[:,:,:,2] = self.img[:,:,:,2]
				self.img_aux[:,:,:,3] = 255
				self.img = self.img_aux

			elif( self.img[0,0,0,:].size == 4 ):
				self.img_nchannels = 4
		
		# SETTING THE OPENCL IMAGE OBJECTS, WITHOUT THE COPY
		self.imgFormat = cl.ImageFormat(self.img_channel_order_cl, self.img_dtype_cl)
		self.img_in_cl = cl.Image(self.ctx, self.mf.READ_ONLY, self.imgFormat, self.img_shape)
		self.img_out_cl = cl.Image(self.ctx, self.mf.WRITE_ONLY, self.imgFormat, self.img_shape)

		# COPYING IMAGE NDARRAY TO OPENCL IMAGE OBJECT
		cl.enqueue_copy(self.queue, self.img_in_cl, self.img, origin=self.img_origin, region=self.img_region, is_blocking=True)

	def execute(self, outputpath):
		# EXECUTING KERNEL WITH THE IMAGES
		print("Executing kernel")
		self.pgr.vglCl3dBlurSq3(self.queue, self.img_in_cl.shape, None, self.img_in_cl, self.img_out_cl)

		# CREATING BUFFER TO GET IMAGE FROM DEVICE
		if( self.img_nchannels == 1 ):
			self.buffer = np.zeros(self.img_shape[2]*self.img_shape[1]*self.img_shape[0], self.img_dtype)
		elif( self.img_nchannels == 3 or self.img_nchannels == 4 ):
			self.buffer = np.zeros(self.img_shape[2]*self.img_shape[1]*self.img_shape[0]*4, self.img_dtype)

		# COPYING IMAGE FROM OPENCL IMAGE OBJECT TO NDARRAY
		cl.enqueue_copy(self.queue, self.buffer, self.img_out_cl, origin=self.img_origin, region=self.img_region, pitches=self.img_pitch, is_blocking=True)

		# TURNING BUFFER INTO A NDARRAY WITH SHAPE SENDED TO THE DEVICE 
		# AND IMG_OUT NDARRAY, THAT HAS THE CORRECT SHAPE TO SAVE IMAGE (AS THE READED IMAGE IS RGB)
		if( self.img_nchannels == 1 ):
			self.buffer = np.frombuffer( self.buffer, self.img_dtype ).reshape( self.img_shape[2], self.img_shape[1], self.img_shape[0] )
		elif( self.img_nchannels == 2 or self.img_nchannels == 3 or self.img_nchannels == 4 ):
			self.buffer = np.frombuffer( self.buffer, self.img_dtype ).reshape( self.img_shape[2], self.img_shape[1], self.img_shape[0], 4 )
			self.img_out = np.empty( (self.img_shape[2], self.img_shape[1], self.img_shape[0], self.img_nchannels), dtype=self.img_dtype )

		# PREPARING TO SAVE THE IMAGE
		if( self.img_nchannels == 1 ):
			# GRAY-SHADES IMAGE
			self.img_out = self.buffer

		elif( self.img_nchannels == 2 ):
			# TWO COLOR CHANNEL IMAGE
			self.img_out = self.buffer
		
		elif( self.img_nchannels == 3 ):
			# THEN WAS ADDED A ALPHA CHANNEL TO IT.
			# NOW, THIS ALPHA CHANNEL WILL BE REMOVED AND THEN THE IMAGE WILL BE SAVED.
			self.img_out[:,:,:,0] = self.buffer[:,:,:,0]
			self.img_out[:,:,:,1] = self.buffer[:,:,:,1]
			self.img_out[:,:,:,2] = self.buffer[:,:,:,2]

		elif( self.img_nchannels == 4 ):
			# THEN ITS WAS A RGBA IMAGE, AND DO NOT NEED ALTERATIONS IN IT.
			self.img_out = self.buffer

		# SAVING IMAGE
		io.imsave(outputpath, self.img_out, plugin='tifffile')

CLPath = "../CL/vglCl3dBlurSq3.cl"
inPath = sys.argv[1]
ouPath = sys.argv[2] 

process = vgl()
process.loadCL(CLPath)
process.loadImage(inPath)
process.execute(ouPath)
