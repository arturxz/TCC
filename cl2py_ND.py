# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

# TO INFER TYPE TO THE VARIABLE
from typing import Union

#TO WORK WITH MAIN
import numpy as np
import sys

class cl2py_ND:
	def __init__(self, cl_ctx=None):
		# PYTHON-EXCLUSIVE VARIABLES
		self.cl_ctx: Union[None, vl.opencl_context] = cl_ctx

		# COMMON VARIABLES. self.ocl IS EQUIVALENT TO cl.
		self._program: Union[None, cl.Program] = None
		self.ocl: Union[None, vl.VglClContext] = None

		if( self.cl_ctx is None ):
			vl.vglClInit()
			self.ocl = vl.get_ocl()
			self.cl_ctx = vl.get_ocl_context()
		else:
			self.ocl = cl_ctx.get_vglClContext_attributes()

	def load_kernel(self, filepath, kernelname):
		print("load_kernel: Loading OpenCL Kernel")
		kernel_file = None

		try:
			kernel_file = open(filepath, "r")
		except FileNotFoundError as fnf:
			print("load_kernel: Error: Kernel File not found. Filepath:", filepath)    
			print(str(fnf))
			exit()
		except Exception as e:
			print("load_kernel: Error: A unexpected exception was thrown while trying to open kernel file. Filepath:", filepath)
			print(str(e))
			exit()

		if(self._program is None):
			print("::Building Kernel")
			self.cl_ctx.load_headers(filepath)
			self._program = cl.Program(self.ocl.context , kernel_file.read())
			self._program.build(options=self.cl_ctx.get_build_options())
		elif( (kernelname in self._program.kernel_names) ):
			print("::Kernel already builded. Going to next step...")
		else:
			print("::Building Kernel")
			self.cl_ctx.load_headers(filepath)
			self._program = cl.Program(self.ocl.context(), kernel_file.read())
			self._program.build(options=self.cl_ctx.get_build_options())
		kernel_file.close()

	def copy_into_byte_array(self, value, byte_array, offset):
		for iterator, byte in enumerate( value.tobytes() ):
			byte_array[iterator+offset] = byte

	def vglClNdConvolution(self, img_input, img_output, window):

		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdConvolution: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		elif( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdConvolution: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()
		else:
			if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
				exit()
			elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
				exit()
			elif( not isinstance(window, vl.VglStrEl) ):
				print("vglClNdConvolution: Error: window is not a VglStrEl object. aborting execution.")
				exit()
			else:
				self.load_kernel("../CL_ND/vglClNdConvolution.cl", "vglClNdConvolution")
				kernel_run = self._program.vglClNdConvolution

				# HERE, THE IMAGE STRUCTURE WILL BE BUILDED.
				struct_sizes = vl.struct_sizes()
				struct_sizes = struct_sizes.get_struct_sizes()

				image_cl_strel = window.asVglClStrEl()
				image_cl_shape = img_input.getVglShape().asVglClShape()

				strel_obj = np.zeros(struct_sizes[0], np.uint8)
				shape_obj = np.zeros(struct_sizes[6], np.uint8)

				# COPYING DATA AS BYTES TO HOST BUFFER
				self.copy_into_byte_array(image_cl_strel.data,	strel_obj, struct_sizes[1])
				self.copy_into_byte_array(image_cl_strel.shape,	strel_obj, struct_sizes[2])
				self.copy_into_byte_array(image_cl_strel.offset,strel_obj, struct_sizes[3])
				self.copy_into_byte_array(image_cl_strel.ndim,	strel_obj, struct_sizes[4])
				self.copy_into_byte_array(image_cl_strel.size,	strel_obj, struct_sizes[5])

				self.copy_into_byte_array(image_cl_shape.ndim,	shape_obj, struct_sizes[7])
				self.copy_into_byte_array(image_cl_shape.shape,	shape_obj, struct_sizes[8])
				self.copy_into_byte_array(image_cl_shape.offset,shape_obj, struct_sizes[9])
				self.copy_into_byte_array(image_cl_shape.size,	shape_obj, struct_sizes[10])

				# CREATING OPENCL BUFFER TO VglStrEl and VglShape
				mobj_window = cl.Buffer(self.ocl.context, cl.mem_flags.READ_ONLY, strel_obj.nbytes)
				mobj_img_shape = cl.Buffer(self.ocl.context, cl.mem_flags.READ_ONLY, shape_obj.nbytes)

				cl.enqueue_copy(self.ocl.commandQueue, mobj_window, strel_obj.tobytes(), is_blocking=True)
				cl.enqueue_copy(self.ocl.commandQueue, mobj_img_shape, shape_obj.tobytes(), is_blocking=True)

				# SETTING ARGUMENTS
				kernel_run.set_arg(0, img_input.get_oclPtr())
				kernel_run.set_arg(1, img_output.get_oclPtr())
				kernel_run.set_arg(2, mobj_img_shape)
				kernel_run.set_arg(3, mobj_window)
				
				# ENQUEUEING KERNEL EXECUTION
				ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
				print(ev)

				vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClNdDilate(self, img_input, img_output, window):

		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdDilate: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		elif( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdDilate: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()
		else:
			if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
				exit()
			elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
				exit()
			elif( not isinstance(window, vl.VglStrEl) ):
				print("vglClNdDilate: Error: window is not a VglStrEl object. aborting execution.")
				exit()
			else:
				self.load_kernel("../CL_ND/vglClNdDilate.cl", "vglClNdDilate")
				kernel_run = self._program.vglClNdDilate

				# HERE, THE IMAGE STRUCTURE WILL BE BUILDED.
				struct_sizes = vl.struct_sizes()
				struct_sizes = struct_sizes.get_struct_sizes()

				image_cl_strel = window.asVglClStrEl()
				image_cl_shape = img_input.getVglShape().asVglClShape()

				strel_obj = np.zeros(struct_sizes[0], np.uint8)
				shape_obj = np.zeros(struct_sizes[6], np.uint8)

				# COPYING DATA AS BYTES TO HOST BUFFER
				self.copy_into_byte_array(image_cl_strel.data,	strel_obj, struct_sizes[1])
				self.copy_into_byte_array(image_cl_strel.shape,	strel_obj, struct_sizes[2])
				self.copy_into_byte_array(image_cl_strel.offset,strel_obj, struct_sizes[3])
				self.copy_into_byte_array(image_cl_strel.ndim,	strel_obj, struct_sizes[4])
				self.copy_into_byte_array(image_cl_strel.size,	strel_obj, struct_sizes[5])

				self.copy_into_byte_array(image_cl_shape.ndim,	shape_obj, struct_sizes[7])
				self.copy_into_byte_array(image_cl_shape.shape,	shape_obj, struct_sizes[8])
				self.copy_into_byte_array(image_cl_shape.offset,shape_obj, struct_sizes[9])
				self.copy_into_byte_array(image_cl_shape.size,	shape_obj, struct_sizes[10])

				# CREATING OPENCL BUFFER TO VglStrEl and VglShape
				mobj_window = cl.Buffer(self.ocl.context, cl.mem_flags.READ_ONLY, strel_obj.nbytes)
				mobj_img_shape = cl.Buffer(self.ocl.context, cl.mem_flags.READ_ONLY, shape_obj.nbytes)

				cl.enqueue_copy(self.ocl.commandQueue, mobj_window, strel_obj.tobytes(), is_blocking=True)
				cl.enqueue_copy(self.ocl.commandQueue, mobj_img_shape, shape_obj.tobytes(), is_blocking=True)

				# SETTING ARGUMENTS
				kernel_run.set_arg(0, img_input.get_oclPtr())
				kernel_run.set_arg(1, img_output.get_oclPtr())
				kernel_run.set_arg(2, mobj_img_shape)
				kernel_run.set_arg(3, mobj_window)
				
				# ENQUEUEING KERNEL EXECUTION
				ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
				print(ev)

				vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	
	def vglClNdCopy(self, img_input, img_output):

		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		elif( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()
		else:
			if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
				exit()
			elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
				exit()
			else:
				self.load_kernel("../CL_ND/vglClNdCopy.cl", "vglClNdCopy")
				kernel_run = self._program.vglClNdCopy

				kernel_run.set_arg(0, img_input.get_oclPtr())
				kernel_run.set_arg(1, img_output.get_oclPtr())
				
				ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
				print(ev)

				vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())


"""
	HERE FOLLOWS THE KERNEL CALLS
"""
if __name__ == "__main__":
	
	wrp = cl2py_ND()

	# INPUT IMAGE
	img_input = vl.VglImage(sys.argv[1], vl.VGL_IMAGE_2D_IMAGE(), vl.IMAGE_ND_ARRAY())
	vl.vglLoadImage(img_input)
	vl.vglClUpload(img_input)

	# OUTPUT IMAGE
	img_output = vl.create_blank_image_as(img_input)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
	vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())

	# STRUCTURANT ELEMENT
	window = vl.VglStrEl()
	window.constructorFromTypeNdim(vl.VGL_STREL_CROSS(), 2)

	#wrp.vglClNdCopy(img_input, img_output)
	#wrp.vglClNdConvolution(img_input, img_output, window)
	wrp.vglClNdDilate(img_input, img_output, window)

	# SAVING IMAGE
	vl.vglClDownload(img_output)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage(sys.argv[2], img_output)

	wrp = None
	img_input = None
	img_output = None
	window = None
