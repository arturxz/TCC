from skimage import io
import pyopencl as cl
import numpy as np

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

class StructSizes:
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
		self.filepath = "get_struct_sizes.cl"

		self.loadCL()
		self.execute()

	# THIS FUNCTION WILL LOAD THE KERNEL FILE
	# AND BUILD IT IF NECESSARY.
	def loadCL(self):
		print("Loading OpenCL Kernel")
		self.kernel_file = open(self.filepath, "r")

		if ((self.builded == False)):
			print("Building OpenCL kernel")
			self.pgr = cl.Program(self.ctx, self.kernel_file.read())
			self.pgr.build()
			self.kernel_file.close()
			self.builded = True
		else:
			print("Kernel already builded. Going to next step...")

	def execute(self):
		# CREATING NUMPY ARRAY
		self.struct_sizes_host = np.zeros(11, np.uint32)
		print(self.struct_sizes_host)

		self.mf = cl.mem_flags
		self.struct_sizes_device = cl.Buffer( self.ctx, self.mf.READ_ONLY, self.struct_sizes_host.nbytes )

		# EXECUTING KERNEL WITH THE IMAGES
		print("Executing kernel")
		self.pgr.get_struct_sizes(self.queue, self.struct_sizes_host.shape, None, self.struct_sizes_device).wait()

		cl.enqueue_copy(self.queue, self.struct_sizes_host, self.struct_sizes_device, is_blocking=True)
		#print(self.struct_sizes_host)

	def get_struct_sizes(self):
		return self.struct_sizes_host