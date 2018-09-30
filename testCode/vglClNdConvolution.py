# IMAGE MANIPULATION LIBRARYS
from skimage import io
import matplotlib.pyplot as mp
import numpy as np

# OPENCL LIBRARYS
import pyopencl as cl
import pyopencl.tools
import pyopencl.array as clarray

# VGL LIBRARYS
from vglImage import *
from vglStrEl import *
import vglConst as vc
from structSizes import *

# SYSTEM LIBRARYS
import sys, glob, os

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
		while( buildDir ):
			for file in glob.glob(buildDir+"/*.h"):
				print(file)
				self.pgr = cl.Program(self.ctx, open(file, "r"))
			
			buildDir = self.getDir(buildDir)
		#for file in glob.glob(buildDir+"/*.h"):
		#	self.pgr = cl.Program(self.ctx, open(file, "r"))

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
		
		self.vglimage = VglImage(imgpath)
		
		mf = cl.mem_flags
		self.vglimage.vglNdImageUpload(self.ctx, self.queue)
		self.img_out_cl = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.vglimage.get_host_image().nbytes)

		self.makeStructures()

	def makeStructures(self):
		print("Making Structures")
		mf = cl.mem_flags

		ss = StructSizes()
		ss = ss.get_struct_sizes()

		# MAKING STRUCTURING ELEMENT
		self.strEl = VglStrEl()
		self.strEl.constructorFromTypeNdim(vc.VGL_STREL_CROSS(), 2)
		
		image_cl_strel = self.strEl.asVglClStrEl()
		image_cl_shape = self.vglimage.getVglShape().asVglClShape()

		vgl_strel_obj = np.zeros(ss[0], np.uint8)
		vgl_shape_obj = np.zeros(ss[6], np.uint8)

		# COPYING DATA AS BYTES TO HOST BUFFER
		self.copy_into_byte_array(image_cl_strel.data,  vgl_strel_obj, ss[1])
		self.copy_into_byte_array(image_cl_strel.shape, vgl_strel_obj, ss[2])
		self.copy_into_byte_array(image_cl_strel.offset,vgl_strel_obj, ss[3])
		self.copy_into_byte_array(image_cl_strel.ndim,  vgl_strel_obj, ss[4])
		self.copy_into_byte_array(image_cl_strel.size,  vgl_strel_obj, ss[5])

		self.copy_into_byte_array(image_cl_shape.ndim,  vgl_shape_obj, ss[7])
		self.copy_into_byte_array(image_cl_shape.shape, vgl_shape_obj, ss[8])
		self.copy_into_byte_array(image_cl_shape.offset,vgl_shape_obj, ss[9])
		self.copy_into_byte_array(image_cl_shape.size, vgl_shape_obj, ss[10])

		print("########## IN PYTHON ##########")
		print("Shape ndim:",  image_cl_shape.ndim)
		print("Shape shape:", image_cl_shape.shape)
		print("Shape offset:",image_cl_shape.offset)
		print("Shape size:",  image_cl_shape.size)

		print("StrEl data:\n", image_cl_strel.data)
		print("StrEl ndim:", image_cl_strel.ndim)
		print("StrEl shape:", image_cl_strel.shape)
		print("StrEl offset:", image_cl_strel.offset)
		print("StrEl size:", image_cl_strel.size)

		print("########## IN BUFFER ##########")
		print("Shape ndim:",  np.frombuffer( vgl_shape_obj, dtype=np.int32, count=1, offset=ss[7] ) )
		print("Shape shape:", np.frombuffer( vgl_shape_obj, dtype=np.int32, count=vc.VGL_ARR_SHAPE_SIZE(), offset=ss[8] ) )
		print("Shape offset:",np.frombuffer( vgl_shape_obj, dtype=np.int32, count=vc.VGL_ARR_SHAPE_SIZE(), offset=ss[9] ) )
		print("Shape size:",  np.frombuffer( vgl_shape_obj, dtype=np.int32, count=1, offset=ss[10] ) )

		print("StrEl data:\n",np.frombuffer( vgl_strel_obj, dtype=np.float32, count=vc.VGL_ARR_CLSTREL_SIZE(), offset=ss[1] ) )
		print("StrEl ndim:",  np.frombuffer( vgl_strel_obj, dtype=np.int32, count=1, offset=ss[4] ) )
		print("StrEl shape:", np.frombuffer( vgl_strel_obj, dtype=np.int32, count=vc.VGL_ARR_SHAPE_SIZE(), offset=ss[2] ) )
		print("StrEl offset:",np.frombuffer( vgl_strel_obj, dtype=np.int32, count=vc.VGL_ARR_SHAPE_SIZE(), offset=ss[3] ) )
		print("StrEl size:",  np.frombuffer( vgl_strel_obj, dtype=np.int32, count=1, offset=ss[5] ) )

		# CREATING DEVICE BUFFER TO HOLD STRUCT DATA
		self.vglstrel_buffer = cl.Buffer(self.ctx, mf.READ_ONLY, vgl_strel_obj.nbytes)
		self.vglshape_buffer = cl.Buffer(self.ctx, mf.READ_ONLY, vgl_shape_obj.nbytes)
				
		# COPYING DATA FROM HOST TO DEVICE
		cl.enqueue_copy(self.queue, self.vglstrel_buffer, vgl_strel_obj.tobytes(), is_blocking=True)
		cl.enqueue_copy(self.queue, self.vglshape_buffer, vgl_shape_obj.tobytes(), is_blocking=True)

	def copy_into_byte_array(self, value, byte_array, offset):
		for iterator, byte in enumerate( value.tobytes() ):
			byte_array[iterator+offset] = byte
		
	def execute(self, outputpath):
		# EXECUTING KERNEL WITH THE IMAGES
		print("Executing kernel")
		
		self.pgr.testprobe(self.queue,
						   self.vglimage.get_host_image().shape,
						   None, 
						   self.vglimage.get_device_image(), 
						   self.img_out_cl,
						   self.vglshape_buffer,
						   self.vglstrel_buffer).wait()
		
		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.vglNdImageDownload(self.ctx, self.queue)
		self.vglimage.img_save(outputpath)

CLPath = "../../CL_ND/testprobe.cl"
inPath = sys.argv[1]
ouPath = sys.argv[2] 

process = vgl()
process.loadCL(CLPath)
process.loadImage(inPath)
process.execute(ouPath)
