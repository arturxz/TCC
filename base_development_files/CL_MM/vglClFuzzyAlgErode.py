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
		
		self.vglimage = VglImage(imgpath)
		if( self.vglimage.getVglShape().getNChannels() == 3 ):
			self.vglimage.rgb_to_rgba()

		self.vglimage.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage.get_similar_device_image_object(self.ctx, self.queue)

	def loadWindow(self):
		mf = cl.mem_flags

		self.arr_window = np.zeros((3,3), np.float32)
		self.arr_window[0,0] = 1
		self.arr_window[0,2] = 1
		self.arr_window[2,0] = 1
		self.arr_window[2,2] = 1
		self.window_x = self.arr_window.shape[0]
		self.window_y = self.arr_window.shape[1]
		self.arr_window_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.arr_window)

	def execute(self, outputpath):
		print("Processing")
		self.pgr.vglClFuzzyAlgErode(self.queue, 
									self.vglimage.get_device_image().shape, 
									None, 
									self.vglimage.get_device_image(),
									self.img_out_cl,
									self.arr_window_cl,
									np.uint32(self.window_x),
									np.uint32(self.window_y))
		
		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)
		if( self.vglimage.getVglShape().getNChannels() == 4 ):
			self.vglimage.rgba_to_rgb()
		self.vglimage.img_save(outputpath)

CLPath = "../CL_MM/vglClFuzzyAlgErode.cl"
inPath = sys.argv[1]
ouPath = sys.argv[2]

process = vgl(CLPath)
process.loadCL(CLPath)
process.loadImage(inPath)
process.loadWindow()
process.execute(ouPath)