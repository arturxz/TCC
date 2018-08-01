# IMAGE MANIPULATION LIBRARYS
from skimage import io
import matplotlib.pyplot as mp
import numpy as np

# OPENCL LIBRARYS
import pyopencl as cl
import pyopencl.tools
import pyopencl.array as clarray

# VGL LIBRARYS
from vglShape import *
from vglStrEl import *
import vglConst as vc
from structSizes import *

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
		buildDir = self.getDir(filepath)

		self.build_options = ""
		self.build_options = self.build_options + "-I "+buildDir
		self.build_options = self.build_options + " -D VGL_SHAPE_NCHANNELS={0}".format(vc.VGL_SHAPE_NCHANNELS())
		self.build_options = self.build_options + " -D VGL_SHAPE_WIDTH={0}".format(vc.VGL_SHAPE_WIDTH())
		self.build_options = self.build_options + " -D VGL_SHAPE_HEIGHT={0}".format(vc.VGL_SHAPE_HEIGHT())
		self.build_options = self.build_options + " -D VGL_SHAPE_LENGTH={0}".format(vc.VGL_SHAPE_LENGTH())
		self.build_options = self.build_options + " -D VGL_MAX_DIM={0}".format(vc.VGL_MAX_DIM())
		self.build_options = self.build_options + " -D VGL_ARR_SHAPE_SIZE={0}".format(vc.VGL_ARR_SHAPE_SIZE())
		self.build_options = self.build_options + " -D VGL_ARR_CLSTREL_SIZE={0}".format(vc.VGL_ARR_CLSTREL_SIZE())
		self.build_options = self.build_options + " -D VGL_STREL_CUBE={0}".format(vc.VGL_STREL_CUBE())
		self.build_options = self.build_options + " -D VGL_STREL_CROSS={0}".format(vc.VGL_STREL_CROSS())
		self.build_options = self.build_options + " -D VGL_STREL_GAUSS={0}".format(vc.VGL_STREL_GAUSS())
		self.build_options = self.build_options + " -D VGL_STREL_MEAN={0}".format(vc.VGL_STREL_MEAN())

		#print("Build Options:\n", self.build_options)

		# READING THE HEADER FILES BEFORE COMPILING THE KERNEL
		for file in glob.glob(buildDir+"/*.h"):
			self.pgr = cl.Program(self.ctx, open(file, "r"))

		if ((self.builded == False)):
			self.pgr = cl.Program(self.ctx, self.kernel_file.read())
			self.pgr.build(options=self.build_options)
			#self.pgr.build()
			self.builded = True
		else:
			print("Kernel already builded. Going to next step...")

		self.kernel_file.close()
		#print("Kernel", self.pgr.get_info(cl.program_info.KERNEL_NAMES), "compiled.")

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
		ss = StructSizes()
		ss = ss.get_struct_sizes()
		print("####GPU RESPONSE####")
		print("tam 1",ss[0])
		print("tam 2",ss[6])

		vgl_strel_obj = np.zeros(ss[0], np.uint8)
		vgl_shape_obj = np.zeros(ss[6], np.uint8)

		print("siz 1", vgl_strel_obj.nbytes)
		print("siz 2", vgl_shape_obj.nbytes)

		# COPYING DATA AS BYTES TO HOST BUFFER
		self.copy_into_byte_array(self.vglClStrEl.data, vgl_strel_obj, ss[1])
		self.copy_into_byte_array(self.vglClStrEl.ndim, vgl_strel_obj, ss[2])
		self.copy_into_byte_array(self.vglClStrEl.shape, vgl_strel_obj, ss[3])
		self.copy_into_byte_array(self.vglClStrEl.offset, vgl_strel_obj, ss[4])
		self.copy_into_byte_array(self.vglClStrEl.size, vgl_strel_obj, ss[5])

		self.copy_into_byte_array(self.vglClShape.ndim, vgl_shape_obj, ss[7])
		self.copy_into_byte_array(self.vglClShape.shape, vgl_shape_obj, ss[8])
		self.copy_into_byte_array(self.vglClShape.offset, vgl_shape_obj, ss[9])
		self.copy_into_byte_array(self.vglClShape.size, vgl_shape_obj, ss[10])

		# CREATING DEVICE BUFFER TO HOLD STRUCT DATA
		self.vglstrel_buffer = cl.Buffer(self.ctx, self.mf.READ_ONLY, vgl_strel_obj.nbytes)
		self.vglshape_buffer = cl.Buffer(self.ctx, self.mf.READ_ONLY, vgl_shape_obj.nbytes)
		
		# COPYING DATA FROM HOST TO DEVICE
		cl.enqueue_copy(self.queue, self.vglstrel_buffer, vgl_strel_obj, is_blocking=True)
		cl.enqueue_copy(self.queue, self.vglshape_buffer, vgl_shape_obj, is_blocking=True)

	def copy_into_byte_array(self, value, byte_array, offset):
		for i,b in enumerate( value.tobytes() ):
			byte_array[i+offset] = b
		
	def execute(self, outputpath):
		# EXECUTING KERNEL WITH THE IMAGES
		print("Executing kernel")
		
		#buffer_list = [self.img_in_cl, self.img_out_cl, self.vglstrel_buffer, self.vglshape_buffer]
		#self.pgr.vglClNdConvolution.set_args(*buffer_list)
		#self.pgr.vglClNdConvolution.set_scalar_arg_dtypes( [None]*len(buffer_list) )

		self.pgr.vglClNdConvolution(self.queue, self.img_shape, None, self.img_in_cl, 
																	  self.img_out_cl,
																	  self.vglshape_buffer,
																	  self.vglstrel_buffer).wait()

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

CLPath = "../../CL_ND/vglClNdConvolution.cl"
inPath = sys.argv[1]
ouPath = sys.argv[2] 

process = vgl()
process.loadCL(CLPath)
process.loadImage(inPath)
process.loadVglObjects()
process.execute(ouPath)
