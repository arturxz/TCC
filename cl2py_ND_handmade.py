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
		self.ocl: Union[None, vl.VglClContext] = None

		if( self.cl_ctx is None ):
			vl.vglClInit()
			self.ocl = vl.get_ocl()
			self.cl_ctx = vl.get_ocl_context()
		else:
			self.ocl = cl_ctx.get_vglClContext_attributes()

	def vglClNdCopy(self, img_input, img_output):
		print("# Running vglClNdCopy")
		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL_ND/vglClNdCopy.cl", "vglClNdCopy")
		kernel_run = _program.vglClNdCopy

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())

		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClNdConvolution(self, img_input, img_output, window):
		print("# Running vglClNdConvolution")
		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdConvolution: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdConvolution: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		if( not isinstance(window, vl.VglStrEl) ):
			print("vglClNdConvolution: Error: window is not a VglStrEl object. aborting execution.")
			exit()

		_program = self.cl_ctx.get_compiled_kernel("../CL_ND/vglClNdConvolution.cl", "vglClNdConvolution")
		kernel_run = _program.vglClNdConvolution

		# CREATING OPENCL BUFFER TO VglStrEl and VglShape
		mobj_window = window.get_asVglClStrEl_buffer()
		mobj_img_shape = img_input.getVglShape().get_asVglClShape_buffer()

		# SETTING ARGUMENTS
		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, mobj_img_shape)
		kernel_run.set_arg(3, mobj_window)
				
		# ENQUEUEING KERNEL EXECUTION
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
		
		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClNdDilate(self, img_input, img_output, window):
		print("# Running vglClNdDilate")
		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdDilate: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdDilate: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		if( not isinstance(window, vl.VglStrEl) ):
			print("vglClNdDilate: Error: window is not a VglStrEl object. aborting execution.")
			exit()
		
		_program = self.cl_ctx.get_compiled_kernel("../CL_ND/vglClNdDilate.cl", "vglClNdDilate")
		kernel_run = _program.vglClNdDilate

		# CREATING OPENCL BUFFER TO VglStrEl and VglShape
		mobj_window = window.get_asVglClStrEl_buffer()
		mobj_img_shape = img_input.getVglShape().get_asVglClShape_buffer()

		# SETTING ARGUMENTS
		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, mobj_img_shape)
		kernel_run.set_arg(3, mobj_window)
				
		# ENQUEUEING KERNEL EXECUTION
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
		
		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClNdErode(self, img_input, img_output, window):
		print("# Running vglClNdErode")
		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdErode: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdErode: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		if( not isinstance(window, vl.VglStrEl) ):
			print("vglClNdErode: Error: window is not a VglStrEl object. aborting execution.")
			exit()

		_program = self.cl_ctx.get_compiled_kernel("../CL_ND/vglClNdErode.cl", "vglClNdErode")
		kernel_run = _program.vglClNdErode

		# CREATING OPENCL BUFFER TO VglStrEl and VglShape
		mobj_window = window.get_asVglClStrEl_buffer()
		mobj_img_shape = img_input.getVglShape().get_asVglClShape_buffer()

		# SETTING ARGUMENTS
		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, mobj_img_shape)
		kernel_run.set_arg(3, mobj_window)
				
		# ENQUEUEING KERNEL EXECUTION
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
		
		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClNdNot(self, img_input, img_output):
		print("# Running vglClNdNot")
		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdNot: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdNot: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL_ND/vglClNdNot.cl", "vglClNdNot")
		kernel_run = _program.vglClNdNot

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
				
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
		
		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClNdThreshold(self, img_input, img_output, thresh, top=255):
		print("# Running vglClNdThreshold")
		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdThreshold: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdThreshold: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		if( not isinstance(thresh, np.uint8) ):
			print("vglClNdThreshold: Warning: thresh not np.uint8! Trying to convert...")
			try:
				thresh = np.uint8(thresh)
			except Exception as e:
				print("vglClNdThreshold: Error!! Impossible to convert thresh as a np.uint8 object.")
				print(str(e))
				exit()
		if( not isinstance(top, np.uint8) ):
			print("vglClNdThreshold: Warning: top not np.uint8! Trying to convert...")
			try:
				top = np.uint8(top)
			except Exception as e:
				print("vglClNdThreshold: Error!! Impossible to convert top as a np.uint8 object.")
				print(str(e))
				exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL_ND/vglClNdThreshold.cl", "vglClNdThreshold")
		kernel_run = _program.vglClNdThreshold

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, thresh)
		kernel_run.set_arg(3, top)
				
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
		
		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

"""
	HERE FOLLOWS THE KERNEL CALLS
"""
if __name__ == "__main__":
	
	wrp = cl2py_ND()

	# INPUT IMAGE
	img_input = vl.VglImage("img-1.jpg", None, vl.VGL_IMAGE_2D_IMAGE(), vl.IMAGE_ND_ARRAY())
	vl.vglLoadImage(img_input)
	vl.vglClUpload(img_input)

	# OUTPUT IMAGE
	img_output = vl.create_blank_image_as(img_input)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
	vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())

	# STRUCTURANT ELEMENT
	window = vl.VglStrEl()
	window.constructorFromTypeNdim(vl.VGL_STREL_CROSS(), 2)

	wrp.vglClNdCopy(img_input, img_output)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdCopy.jpg", img_output)

	wrp.vglClNdConvolution(img_input, img_output, window)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdConvolution.jpg", img_output)
	
	wrp.vglClNdDilate(img_input, img_output, window)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdDilate.jpg", img_output)

	wrp.vglClNdErode(img_input, img_output, window)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdErode.jpg", img_output)

	wrp.vglClNdNot(img_input, img_output)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdNot.jpg", img_output)

	wrp.vglClNdThreshold(img_input, img_output, np.uint8(120), np.uint8(190))
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdThreshold.jpg", img_output)

	wrp = None
	img_input = None
	img_output = None
	window = None
