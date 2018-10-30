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
		
		self.vglimage = VglImage(imgpath)
		
		mf = cl.mem_flags
		self.vglimage.vglNdImageUpload(self.ctx, self.queue)
		self.img_out_cl = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.vglimage.get_host_image().nbytes)
		
	def execute(self, outputpath):
		# EXECUTING KERNEL WITH THE IMAGES
		print("Executing kernel")
		self.pgr.vglClNdNot(self.queue,
							self.vglimage.get_host_image().shape,
							None,
							self.vglimage.get_device_image(),
							self.img_out_cl).wait()
		
		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.vglNdImageDownload(self.ctx, self.queue)
		self.vglimage.img_save(outputpath)


CLPath = "../../CL_ND/vglClNdNot.cl"
inPath = sys.argv[1]
ouPath = sys.argv[2] 

process = vgl()
process.loadCL(CLPath)
process.loadImage(inPath)
process.execute(ouPath)