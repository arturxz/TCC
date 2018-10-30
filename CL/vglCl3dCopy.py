# IMAGE MANIPULATION LIBRARYS
from skimage import io
import numpy as np

# OPENCL LIBRARYS
import pyopencl as cl

# VGL LIBRARYS
from vglImage import *
from vglStrEl import *
from structSizes import *
from vglOclContext import *
import vglConst as vc

class vgl:
	# THE vgl CONSTRUCTOR CREATES A NEW CONTEXT
	# AND INITIATES THE QUEUE, ADDING QUE CONTEXT TO IT.
	def __init__(self, filepath):
		print("Starting OpenCL")
		self.ocl_ctx = VglOclContext()
		self.ocl_ctx.load_headers(filepath)
		self.ctx = self.ocl_ctx.get_context()
		self.queue = self.ocl_ctx.get_queue()
		self.builded = False

	# THIS FUNCTION WILL LOAD THE KERNEL FILE
	# AND BUILD IT IF NECESSARY.
	def loadCL(self, filepath):
		print("Loading OpenCL Kernel")
		self.kernel_file = open(filepath, "r")

		if ((self.builded == False)):
			print("::Building Kernel")
			self.pgr = cl.Program(self.ctx, self.kernel_file.read())
			self.pgr.build(options=self.ocl_ctx.get_build_options())
			self.builded = True
		else:
			print("::Kernel already builded. Going to next step...")

		self.kernel_file.close()

	def loadImage(self, imgpath):
		print("Opening image to be processed")
		
		self.vglimage = VglImage(imgpath, vc.VGL_IMAGE_3D_IMAGE())
		self.vglimage.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage.get_similar_device_image_object(self.ctx, self.queue)
	
	def execute(self, outputpath):
		# EXECUTING KERNEL WITH THE IMAGES
		print("Executing kernel")
		self.pgr.vglCl3dCopy(self.queue, 
							 self.img_out_cl.shape, 
							 None, 
							 self.vglimage.get_device_image(),
							 self.img_out_cl)

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)
		self.vglimage.img_save(outputpath)

CLPath = "../CL/vglCl3dCopy.cl"
inPath = sys.argv[1]
ouPath = sys.argv[2] 

process = vgl(CLPath)
process.loadCL(CLPath)
process.loadImage(inPath)
process.execute(ouPath)