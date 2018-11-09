# IMAGE MANIPULATION LIBRARYS
from skimage import io
import numpy as np
import sys

# OPENCL LIBRARYS
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

class Wrapper:
	def __init__(self, filepath):
		self.filepath = filepath
		self.ocl_ctx = vl.VglOclContext()
		self.ocl_ctx.load_headers(filepath)
		self.ctx = self.ocl_ctx.get_context()
		self.queue = self.ocl_ctx.get_queue()
		self.builded = False
	
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
	
	"""
		HERE FOLLOWS THE IMAGE LOADING METHODS
	"""
	# LOADING INTO A 3D IMAGE OBJECT
	def loadImage3D(self, imgpath):
		print("Opening image to be processed")
		
		self.vglimage = vl.VglImage(imgpath, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage.get_similar_device_image_object(self.ctx, self.queue)
	
	# LOADING INTO A 2D IMAGE OBJECT
	def loadImage2D(self, imgpath):
		print("Opening image to be processed")
		
		self.vglimage = vl.VglImage(imgpath, vl.VGL_IMAGE_2D_IMAGE())
		self.vglimage.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage.get_similar_device_image_object(self.ctx, self.queue)
	
	# LOADING INTO A NDARRAY IMAGE
	def loadImageND(self, imgpath):
		print("Opening image to be processed")
		
		self.vglimage = vl.VglImage(imgpath)
		
		mf = cl.mem_flags
		self.vglimage.vglNdImageUpload(self.ctx, self.queue)
		self.img_out_cl = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.vglimage.get_host_image().nbytes)
	
	"""
		HERE FOLLOWS THE METHODS THAT STRUCTURE THE 
		VGLSHAPE AND VGLSTREL BUFFERS TO SEND TO OPENCL.
	"""
	def makeStructures(self, strElType, strElDim):
		print("Making Structures")
		mf = cl.mem_flags

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
		self.vglstrel_buffer = cl.Buffer(self.ctx, mf.READ_ONLY, vgl_strel_obj.nbytes)
		self.vglshape_buffer = cl.Buffer(self.ctx, mf.READ_ONLY, vgl_shape_obj.nbytes)
				
		# COPYING DATA FROM HOST TO DEVICE
		cl.enqueue_copy(self.queue, self.vglstrel_buffer, vgl_strel_obj.tobytes(), is_blocking=True)
		cl.enqueue_copy(self.queue, self.vglshape_buffer, vgl_shape_obj.tobytes(), is_blocking=True)

	def copy_into_byte_array(self, value, byte_array, offset):
		for iterator, byte in enumerate( value.tobytes() ):
			byte_array[iterator+offset] = byte

	"""
		HERE FOLLOWS THE IMAGE SAVING METHODS
	"""
	def saveImage(self, outputpath):
		self.vglimage.img_save(outputpath)

	"""
		HERE FOLLOWS THE MODULE CALLS
	"""
	def vglClNdConvolution(self, imgIn, strElType, strElDim):
		self.loadCL(self.filepath)
		self.loadImageND(imgIn)
		self.makeStructures(strElType, strElDim)

		self.pgr.vglClNdConvolution(self.queue,
									self.vglimage.get_host_image().shape,
									None, 
									self.vglimage.get_device_image(), 
									self.img_out_cl,
									self.vglshape_buffer,
									self.vglstrel_buffer).wait()
		
		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.vglNdImageDownload(self.ctx, self.queue)
	
	def vglClNdCopy(self, imgIn):
		self.loadCL(self.filepath)
		self.loadImageND(imgIn)

		self.pgr.vglClNdCopy(self.queue,
							self.vglimage.get_host_image().shape,
							None,
							self.vglimage.get_device_image(),
							self.img_out_cl).wait()
		
		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.vglNdImageDownload(self.ctx, self.queue)

if __name__ == "__main__":
	"""
	wrp = Wrapper("../CL_ND/vglClNdConvolution.cl")

	wrp.vglClNdConvolution(sys.argv[1], vl.VGL_STREL_CROSS(), 2)
	wrp.saveImage(sys.argv[2])
	"""
	wrp = Wrapper("../CL_ND/vglClNdCopy.cl")

	wrp.vglClNdCopy(sys.argv[1])
	wrp.saveImage(sys.argv[2])