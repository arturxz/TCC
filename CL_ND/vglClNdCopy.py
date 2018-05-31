from skimage import io
import matplotlib.pyplot as mp
import pyopencl as cl
import numpy as np
import sys

"""
	img:
		is the input image
	img_size:
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

		# GETTING NDARRAY IMAGE AND DATA ABOUT THE IMAGE
		self.img = io.imread(imgpath)
		self.img_dtype = self.img.dtype
		self.img_ndim = self.img.ndim
		self.img_shape = (self.img.shape[1],self.img.shape[0])

		# GETTING THE DIMENSIONS OF THE IMAGE
		if( self.img_ndim == 2 ):
			# IF THE IMAGE IS 2-DIMENSIONAL, THEN IT IS A SHADES OF GRAY IMAGE
			# AND THE IMAGE TYPE IS LUMINANCE
			self.img_nchannels = 1
		elif( self.img_ndim == 3 ):
			# IF THE IMAGE ARRAY IS 3-DIMENSIONAL, THEN IT HAS MORE THAN 1 COLOR CHANNEL

			if( self.img[0,0,:].size == 2 ):
				# THEN IT CAN BE ANY 2-CHANNEL IMAGE.
				# DON'T HAVE ACCES TO ANY IMAGE LIKE THAT YET.
				self.img_nchannels = 2

			if( self.img[0,0,:].size == 3 ):
				# THEN IT CAN BE ANY 3-CHANNEL IMAGE. IS NEEDED TO ADD THE 4TH CHANNEL TO IT
				self.img_nchannels = 3

		# SETTING THE OPENCL IMAGE OBJECTS, WITHOUT THE COPY
		self.img_in_cl = cl.Buffer(self.ctx, self.mf.READ_ONLY, self.img.nbytes)
		self.img_out_cl = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.img.nbytes)

		# COPYING NDARRAY IMAGE TO OPENCL IMAGE OBJECT
		cl.enqueue_copy(self.queue, self.img_in_cl, self.img, is_blocking=True)
		io.imsave('img_in.jpg', self.img)

	def execute(self, outputpath):
		# EXECUTING KERNEL WITH THE IMAGES
		print("Executing kernel")
		self.pgr.vglClNdCopy(self.queue, self.img_shape, None, self.img_in_cl, self.img_out_cl).wait()

		# CREATING BUFFER TO GET IMAGE FROM DEVICE
		if( self.img_nchannels == 1 ):
			self.buffer = np.zeros(self.img_shape[0]*self.img_shape[1], self.img_dtype)
		elif( self.img_nchannels > 1 ):
			self.buffer = np.zeros(self.img_shape[0]*self.img_shape[1]*self.img_nchannels, self.img_dtype)

		cl.enqueue_copy(self.queue, self.buffer, self.img_in_cl, is_blocking=True)
		
		# TURNING BUFFER INTO A NDARRAY WITH SHAPE SENDED TO THE DEVICE 
		# AND IMG_OUT NDARRAY, THAT HAS THE CORRECT SHAPE TO SAVE IMAGE (AS THE READED IMAGE IS RGB)
		if( self.img_nchannels == 1 ):
			self.buffer = np.frombuffer( self.buffer, self.img_dtype ).reshape( self.img_shape[1], self.img_shape[0] )
		
		elif( self.img_nchannels > 1 ):
			self.buffer = np.frombuffer( self.buffer, self.img_dtype ).reshape( self.img_shape[1], self.img_shape[0], self.img_nchannels )

		self.img_out = self.buffer

		# SAVING IMAGE (USING PIL PLUGIN)
		io.imsave('img_out.jpg', self.buffer)
		io.imsave(outputpath, self.img_out)

CLPath = "../CL_ND/vglClNdCopy.cl"
inPath = sys.argv[1]
ouPath = sys.argv[2]

process = vgl()
process.loadCL(CLPath)
process.loadImage(inPath)
process.execute(ouPath)
