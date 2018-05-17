# IMAGE MANIPULATION LIBRARYS
from skimage import io
import matplotlib.pyplot as mp
import numpy as np

# OPENCL LIBRARYS
import pyopencl as cl
import pyopencl.tools
import pyopencl.array

# VGL LIBRARYS
from vglShape import *
from vglStrEl import *
import vglConst as vc

# SYSTEM LIBRARYS
import sys, glob, os

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

	def getDir(self, filePath):
		size = len(filePath)-1
		bar = -1
		for i in range(0, size):
			if(filePath[i] == '/'):
				bar = i
				i = -1
		return filePath[:bar+1]

	# THIS FUNCTION WILL LOAD THE KERNEL FILE
	# AND BUILD IT IF NECESSARY.
	def loadCL(self, filepath):
		print("Loading OpenCL Kernel")
		self.kernel_file = open(filepath, "r")
		#self.build_options = []
		#self.build_options.append("-I "+self.getDir(filepath))
		#self.build_options.append("-I /home/artur/visiongl/src/CL_ND")

		# READING THE HEADER FILES BEFORE COMPILING THE KERNEL
		for file in glob.glob(self.getDir(filepath)+"/*.h"):
			self.pgr = cl.Program(self.ctx, open(self.getDir(filepath)+"/"+file, "r"))

		if ((self.builded == False)):
			self.pgr = cl.Program(self.ctx, self.kernel_file.read())
			#self.pgr.build(options=self.build_options)
			self.pgr.build()
			self.builded = True
		else:
			print("Kernel already builded. Going to next step...")

		self.kernel_file.close()
		print("Kernel", self.pgr.get_info(cl.program_info.KERNEL_NAMES), "compiled.")

	def loadImage(self, imgpath):
		print("Opening image to be processed")
		self.mf = cl.mem_flags

		# GETTING NDARRAY IMAGE AND DATA ABOUT THE IMAGE
		self.img = io.imread(imgpath)
		self.img_dtype = self.img.dtype
		self.img_ndim = self.img.ndim
		self.img_shape = (self.img.shape[1],self.img.shape[0])
		self.img_origin = (0, 0, 0)
		self.img_region = (self.img_shape[0], self.img_shape[1], 1)

		if( self.img.dtype == np.uint8 ):
			self.img_dtype_cl = cl.channel_type.UNORM_INT8

		# GETTING THE DIMENSIONS OF THE IMAGE
		if( self.img_ndim == 1 ):
			# WHAT TO DO IF THE IMAGE IS 1-DIMENSIONAL
			self.img_channel_order_cl = cl.channel_order.LUMINANCE
		elif( self.img_ndim == 2 ):
			# IF THE IMAGE IS 2-DIMENSIONAL, THEN IT IS A SHADES OF GRAY IMAGE
			# AND THE IMAGE TYPE IS LUMINANCE
			self.img_channel_order_cl = cl.channel_order.LUMINANCE
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

				"""
					DON'T KNOW YET HOW TO DISCOVER IF IT IS RGB, RBG, BGR, GBR, HSV OR SOME OTHER TYPE.
					FOR NOW, THIS PROGRAM WAS JUST TESTED WITH RGB AND SHOULD WORK WITH ITS 3-CHANNEL VARIANTS.
					HERE IS JUST ADDED AN ALPHA CHANNEL FOR THE IMAGE TO BE IN RGBA (BGRA, RBGA, [...], FORMAT)
				"""

				# TURNING INTO RGBA IMAGE
				self.img_aux = np.empty((self.img.shape[0],self.img.shape[1],4), self.img_dtype)
				self.img_aux[:,:,0] = self.img[:,:,0]
				self.img_aux[:,:,1] = self.img[:,:,1]
				self.img_aux[:,:,2] = self.img[:,:,2]
				self.img_aux[:,:,3] = 255

				self.img = self.img_aux
				self.img_channel_order_cl = cl.channel_order.RGBA

		elif( self.img_ndim == 4 ):
			# THEN IT COULD BE ANY IMAGE WITH 4 COLOR CHANNELS
			# DON'T NEED TO DO ENYTHING WITH THE IMAGE
			# SO, JUST MAKING THE IMAGE FORMAT OBJECT
			self.img_channel_order_cl = cl.channel_order.RGBA

		# SETTING THE OPENCL IMAGE OBJECTS, WITHOUT THE COPY
		self.imgFormat = cl.ImageFormat(self.img_channel_order_cl, self.img_dtype_cl)
		self.img_in_cl = cl.Buffer(self.ctx, self.mf.READ_ONLY, self.img.nbytes)
		self.img_out_cl = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, self.img.nbytes)

		# COPYING NDARRAY IMAGE TO OPENCL IMAGE OBJECT
		cl.enqueue_copy(self.queue, self.img_in_cl, self.img.tobytes(), is_blocking=True)

	def loadVglObjects(self):
		self.vglShape = VglShape()
		self.vglShape.constructor2DShape(self.img_nchannels, self.img_shape[1], self.img_shape[0])
		self.vglClShape = self.vglShape.asVglClShape()

		self.strEl = VglStrEl()
		self.strEl.constructorFromTypeNdim(vc.VGL_STREL_GAUSS(), 5) # gaussian blur of size 5
		self.vglClStrEl = self.strEl.asVglClStrEl()

		self.makeStructures()

	def makeStructures(self):
		self.vglClShape_struct = np.dtype([ ("ndim",	np.int32), 
										    ("shape",	np.int32), 
										    ("offset",	np.int32),
										    ("size",	np.int32) ])

		vglClShape_struct, vglClShape_struct_c_decl = cl.tools.match_dtype_to_c_struct(self.ctx.devices[0], "VglClShape", self.vglClShape_struct)
		vglClShape_struct = cl.tools.get_or_register_dtype("VglClShape", vglClShape_struct)
		print(vglClShape_struct)

	def execute(self, outputpath):
		# EXECUTING KERNEL WITH THE IMAGES
		print("Executing kernel")
		self.pgr.vglClNdConvolution(self.queue, self.img_shape, None, self.img_in_cl, 
																	  self.img_out_cl, 
																	  bytes(self.vglShape_cl),
																	  self.vglClStrEl).wait()

		# CREATING BUFFER TO GET IMAGE FROM DEVICE
		if( self.img_nchannels == 1 ):
			self.buffer = np.zeros(self.img_shape[0]*self.img_shape[1], self.img_dtype)
		elif( self.img_nchannels == 3 or self.img_nchannels == 4 ):
			self.buffer = np.zeros(self.img_shape[0]*self.img_shape[1]*4, self.img_dtype)

		cl.enqueue_copy(self.queue, self.buffer, self.img_out_cl, is_blocking=True)
		
		# TURNING BUFFER INTO A NDARRAY WITH SHAPE SENDED TO THE DEVICE 
		# AND IMG_OUT NDARRAY, THAT HAS THE CORRECT SHAPE TO SAVE IMAGE (AS THE READED IMAGE IS RGB)
		if( self.img_nchannels == 1 ):
			self.buffer = np.frombuffer( self.buffer, self.img_dtype ).reshape( self.img_shape[1], self.img_shape[0] )
		
		elif( self.img_nchannels == 3 or self.img_nchannels == 4 ):
			self.buffer = np.frombuffer( self.buffer, self.img_dtype ).reshape( self.img_shape[1], self.img_shape[0], 4 )
			self.img_out = np.empty( (self.buffer.shape[0],self.buffer.shape[1],self.img_nchannels), dtype=self.img_dtype )

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
			self.img_out[:,:,0] = self.buffer[:,:,0]
			self.img_out[:,:,1] = self.buffer[:,:,1]
			self.img_out[:,:,2] = self.buffer[:,:,2]

		elif( self.img_nchannels == 4 ):
			# THEN ITS WAS A RGBA IMAGE, AND DO NOT NEED ALTERATIONS IN IT.
			self.img_out = self.buffer

		# SAVING IMAGE (USING PIL PLUGIN)
		io.imsave(outputpath, self.img_out, plugin='pil')

CLPath = "../CL_ND/vglClNdConvolution.cl"
inPath = sys.argv[1]
ouPath = sys.argv[2] 

process = vgl()
process.loadCL(CLPath)
process.loadImage(inPath)
process.loadVglObjects()
#process.execute(ouPath)
