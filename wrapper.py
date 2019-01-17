# IMAGE MANIPULATION LIBRARYS
from skimage import io
import numpy as np
import sys

# OPENCL LIBRARYS
import pyopencl

# VGL LIBRARYS
import vgl_lib as vl

class Wrapper:
	def __init__(self):
		self.cl_ctx = vl.opencl_context()
		self.ctx = self.cl_ctx.get_context()
		self.queue = self.cl_ctx.get_queue()
		self.pgr = None

		vl.setOcl( self.cl_ctx.get_vglClContext_attributes() )
	
	def loadCL(self, filepath, kernelname):
		print("Loading OpenCL Kernel")
		self.kernel_file = open(filepath, "r")

		if(self.pgr is None):
			print("::Building Kernel")
			self.cl_ctx.load_headers(filepath)
			self.pgr = pyopencl.Program(self.ctx, self.kernel_file.read())
			self.pgr.build(options=self.cl_ctx.get_build_options())
			self.builded = True
		elif( (kernelname in self.pgr.kernel_names) ):
			print("::Kernel already builded. Going to next step...")
		else:
			print("::Building Kernel")
			self.cl_ctx.load_headers(filepath)
			self.pgr = pyopencl.Program(self.ctx, self.kernel_file.read())
			self.pgr.build(options=self.cl_ctx.get_build_options())
			self.builded = True
		self.kernel_file.close()
	
	"""
		HERE FOLLOWS THE IMAGE LOADING METHODS
	"""
	# LOADING INTO A 3D IMAGE OBJECT
	def loadImage3D(self, imgpath):
		print("Opening image to be processed")
		
		self.vglimage = vl.VglImage(imgpath, vl.VGL_IMAGE_3D_IMAGE())
		vl.vglClUpload(self.vglimage)
		self.img_out_cl = self.vglimage.get_similar_oclPtr_object(self.ctx, self.queue)
	
	# LOADING INTO A 2D IMAGE OBJECT
	def loadImage2D(self, imgpath):
		print("Opening image to be processed")
		
		self.vglimage = vl.VglImage(imgpath, vl.VGL_IMAGE_2D_IMAGE())
		vl.vglLoadImage(self.vglimage, imgpath)

		if( self.vglimage.getVglShape().getNChannels() == 3 ):
			vl.rgb_to_rgba(self.vglimage)
	
		vl.vglClUpload(self.vglimage)
		self.img_out_cl = vl.get_similar_oclPtr_object(self.vglimage)

		return self.vglimage
	
	# LOADING INTO A NDARRAY IMAGE
	def loadImageND(self, imgpath):
		print("Opening image to be processed")
		
		self.vglimage = vl.VglImage(imgpath)
		
		mf = pyopencl.mem_flags
		self.vglimage.vglNdImageUpload(self.ctx, self.queue)
		self.img_out_cl = pyopencl.Buffer(self.ctx, mf.WRITE_ONLY, self.vglimage.get_ram_image().nbytes)
	
	"""
		HERE FOLLOWS THE METHODS THAT STRUCTURE THE 
		VGLSHAPE AND VGLSTREL BUFFERS TO OPENCL.
	"""
	def makeStructures(self, strElType, strElDim):
		print("Making Structures")
		mf = pyopencl.mem_flags

		ss = vl.StructSizes()
		ss = ss.get_struct_sizes()

		# MAKING STRUCTURING ELEMENT
		self.strEl = vl.VglStrEl()
		self.strEl.constructorFromTypeNdim(strElType, strElDim)
		
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
		self.vglstrel_buffer = pyopencl.Buffer(self.ctx, mf.READ_ONLY, vgl_strel_obj.nbytes)
		self.vglshape_buffer = pyopencl.Buffer(self.ctx, mf.READ_ONLY, vgl_shape_obj.nbytes)
				
		# COPYING DATA FROM HOST TO DEVICE
		pyopencl.enqueue_copy(self.queue, self.vglstrel_buffer, vgl_strel_obj.tobytes(), is_blocking=True)
		pyopencl.enqueue_copy(self.queue, self.vglshape_buffer, vgl_shape_obj.tobytes(), is_blocking=True)

	def copy_into_byte_array(self, value, byte_array, offset):
		for iterator, byte in enumerate( value.tobytes() ):
			byte_array[iterator+offset] = byte

	"""
		HERE FOLLOWS THE IMAGE SAVING METHOD
	"""
	def saveImage(self, img, outputpath):
		ext = outputpath.split(".")
		ext.reverse()

		vl.vglCheckContext(img, vl.VGL_RAM_CONTEXT())

		if( ext.pop(0).lower() == 'jpg' ):
			if( img.getVglShape().getNChannels() == 4 ):
				vl.rgba_to_rgb(img)
	
		vl.vglSaveImage(outputpath, img)

	def vglClBlurSq3(self, img_input, img_output):
		import pyopencl as cl

		#vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		#vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		self.loadCL("../CL/vglClBlurSq3.cl", "vglClBlurSq3")
		"""
		self.pgr.vglClBlurSq3(self.queue,
							  img_output.get_device_image().shape,
							  None,
							  img_input.get_device_image(), 
							  img_output.get_device_image()).wait()
		"""
		kernel_run = self.pgr.vglClBlurSq3
		#kernel_run.set_args(img_input.get_device_image(), img_output.get_device_image())
		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		ev = cl.enqueue_nd_range_kernel(self.queue, kernel_run, img_output.get_oclPtr().shape, None)
		
		#img_input.set_oclPtr(img_output.get_oclPtr())

"""
	HERE FOLLOWS THE KERNEL CALLS
"""
if __name__ == "__main__":


	"""
		NEXT THING TO DO: 
			MAKE WRAPPER APPEAR LIKE C++ VERSION
	"""

	#wrp = Wrapper()

	
	# vglClBlurSq3
	img_input = vl.VglImage("", vl.VGL_IMAGE_2D_IMAGE())
	vl.vglLoadImage(img_input, sys.argv[1])
	if( img_input.getVglShape().getNChannels() == 3 ):
		vl.rgb_to_rgba(img_input)
	
	vl.vglClUpload(img_input)

	img_output = vl.create_blank_image_as(img_input)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
	vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())
	
	wrp.vglClBlurSq3(img_input, img_output)
	
	vl.vglClDownload(img_output)
	wrp.saveImage(img_output, sys.argv[2])