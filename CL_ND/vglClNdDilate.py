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
		
		self.pgr.vglClNdDilate(self.queue,
							   self.vglimage.get_host_image().shape,
							   None, 
							   self.vglimage.get_device_image(), 
							   self.img_out_cl,
							   self.vglshape_buffer,
							   self.vglstrel_buffer).wait()
		
		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.vglNdImageDownload(self.ctx, self.queue)
		self.vglimage.img_save(outputpath)

#CLPath = "../../CL_ND/testprobe.cl"
CLPath = "../../CL_ND/vglClNdDilate.cl"
inPath = sys.argv[1]
ouPath = sys.argv[2] 

process = vgl(CLPath)
process.loadCL(CLPath)
process.loadImage(inPath)
process.execute(ouPath)
