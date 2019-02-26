# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

# TO INFER TYPE TO THE VARIABLE
from typing import Union

#TO WORK WITH MAIN
import numpy as np
import sys

class cl2py_BIN_ND:
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

	def vglClNdBinDilate(self, img_input, img_output, window):

		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinDilate: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinDilate: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		if( not isinstance(window, vl.VglStrEl) ):
			print("vglClNdBinDilate: Error: window is not a VglStrEl object. aborting execution.")
			exit()
		
		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClNdBinDilate.cl", "vglClNdBinDilate")
		kernel_run = _program.vglClNdBinDilate

		# CREATING OPENCL BUFFER TO VglStrEl and VglShape
		mobj_window = window.get_asVglClStrEl_buffer()
		mobj_img_shape = img_input.getVglShape().get_asVglClShape_buffer()

		# SETTING ARGUMENTS
		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, mobj_img_shape)
		kernel_run.set_arg(3, mobj_window)

		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		worksize = (int(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )
		# ENQUEUEING KERNEL EXECUTION
		#cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
		
		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClNdBinDilatePack(self, img_input, img_output, window):

		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinDilatePack: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinDilatePack: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		if( not isinstance(window, vl.VglStrEl) ):
			print("vglClNdBinDilatePack: Error: window is not a VglStrEl object. aborting execution.")
			exit()
		
		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClNdBinDilatePack.cl", "vglClNdBinDilatePack")
		kernel_run = _program.vglClNdBinDilatePack

		# CREATING OPENCL BUFFER TO VglStrEl and VglShape
		mobj_window = window.get_asVglClStrEl_buffer()
		mobj_img_shape = img_input.getVglShape().get_asVglClShape_buffer()

		# SETTING ARGUMENTS
		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, mobj_img_shape)
		kernel_run.set_arg(3, mobj_window)

		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		worksize = (int(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )
				
		# ENQUEUEING KERNEL EXECUTION
		#cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
		
		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClNdBinErode(self, img_input, img_output, window):

		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinErode: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinErode: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		if( not isinstance(window, vl.VglStrEl) ):
			print("vglClNdBinErode: Error: window is not a VglStrEl object. aborting execution.")
			exit()

		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClNdBinErode.cl", "vglClNdBinErode")
		kernel_run = _program.vglClNdBinErode

		# CREATING OPENCL BUFFER TO VglStrEl and VglShape
		mobj_window = window.get_asVglClStrEl_buffer()
		mobj_img_shape = img_input.getVglShape().get_asVglClShape_buffer()

		# SETTING ARGUMENTS
		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, mobj_img_shape)
		kernel_run.set_arg(3, mobj_window)
				
		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		worksize = (int(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )
				
		# ENQUEUEING KERNEL EXECUTION
		#cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
		
		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClNdBinErodePack(self, img_input, img_output, window):

		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinErodePack: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinErodePack: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		if( not isinstance(window, vl.VglStrEl) ):
			print("vglClNdBinErodePack: Error: window is not a VglStrEl object. aborting execution.")
			exit()

		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClNdBinErodePack.cl", "vglClNdBinErodePack")
		kernel_run = _program.vglClNdBinErodePack

		# CREATING OPENCL BUFFER TO VglStrEl and VglShape
		mobj_window = window.get_asVglClStrEl_buffer()
		mobj_img_shape = img_input.getVglShape().get_asVglClShape_buffer()

		# SETTING ARGUMENTS
		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, mobj_img_shape)
		kernel_run.set_arg(3, mobj_window)
				
		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		worksize = (int(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )
				
		# ENQUEUEING KERNEL EXECUTION
		#cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
		
		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClNdBinMax(self, img_input, img_input2, img_output):

		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinMax: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_input2.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinMax: Error: this function supports only OpenCL data as buffer and img_input2 isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinMax: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		if( not isinstance(window, vl.VglStrEl) ):
			print("vglClNdBinMax: Error: window is not a VglStrEl object. aborting execution.")
			exit()

		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClNdBinMax.cl", "vglClNdBinMax")
		kernel_run = _program.vglClNdBinMax

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_input2.get_oclPtr())
		kernel_run.set_arg(2, img_output.get_oclPtr())
				
		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_input2.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input2.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		worksize = (int(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )
				
		# ENQUEUEING KERNEL EXECUTION
		#cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
		
		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClNdBinMin(self, img_input, img_input2, img_output):

		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinMin: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_input2.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinMin: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinMin: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		if( not isinstance(window, vl.VglStrEl) ):
			print("vglClNdBinMin: Error: window is not a VglStrEl object. aborting execution.")
			exit()

		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClNdBinMin.cl", "vglClNdBinMin")
		kernel_run = _program.vglClNdBinMin

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_input2.get_oclPtr())
		kernel_run.set_arg(2, img_output.get_oclPtr())
		
		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_input2.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input2.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		worksize = (int(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )
				
		# ENQUEUEING KERNEL EXECUTION
		#cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)
		
		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClNdBinNot(self, img_input, img_output):

		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdNot: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdNot: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		if( not isinstance(window, vl.VglStrEl) ):
			print("vglClNdNot: Error: window is not a VglStrEl object. aborting execution.")
			exit()

		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClNdBinNot.cl", "vglClNdBinNot")
		kernel_run = _program.vglClNdBinNot

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
				
		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		worksize = (int(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )
				
		# ENQUEUEING KERNEL EXECUTION
		#cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	"""
	def vglClNdBinRoi(self, img_input, p0, pf):

		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinRoi: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()

		try:
			cl_p0 = cl.Buffer(self.ocl.context, cl.mem_flags.READ_ONLY, p0.nbytes)
			cl.enqueue_copy(self.ocl.commandQueue, cl_p0, p0.tobytes(), is_blocking=True)
			p0 = cl_p0
		except Exception as e:
			print("vglClConvolution: Error!! Impossible to convert p0 to cl.Buffer object.")
			print(str(e))
			exit()

		try:
			cl_pf = cl.Buffer(self.ocl.context, cl.mem_flags.READ_ONLY, pf.nbytes)
			cl.enqueue_copy(self.ocl.commandQueue, cl_pf, pf.tobytes(), is_blocking=True)
			pf = cl_pf
		except Exception as e:
			print("vglClConvolution: Error!! Impossible to convert pf to cl.Buffer object.")
			print(str(e))
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		if( not isinstance(window, vl.VglStrEl) ):
			print("vglClNdBinRoi: Error: window is not a VglStrEl object. aborting execution.")
			exit()

		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClNdBinRoi.cl", "vglClNdBinRoi")
		kernel_run = _program.vglClNdBinRoi

		mobj_img_shape = img_input.getVglShape().get_asVglClShape_buffer()

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, p0)
		kernel_run.set_arg(2, pf)
		kernel_run.set_arg(3, mobj_img_shape)
				
		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		#worksize = (int(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )
				
		# ENQUEUEING KERNEL EXECUTION
		#cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_ipl().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	"""

	def vglClNdBinThreshold(self, img_input, img_output, thresh):

		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinThreshold: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinThreshold: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		if( not isinstance(thresh, np.uint8) ):
			print("vglClNdBinThreshold: Warning: thresh not np.uint8! Trying to convert...")
			try:
				thresh = np.uint8(thresh)
			except Exception as e:
				print("vglClNdBinThreshold: Error!! Impossible to convert thresh as a np.uint8 object.")
				print(str(e))
				exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClNdBinThreshold.cl", "vglClNdBinThreshold")
		kernel_run = _program.vglClNdBinThreshold

		mobj_img_shape_input = img_input.getVglShape().get_asVglClShape_buffer()
		mobj_img_shape_output = img_output.getVglShape().get_asVglClShape_buffer()

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, thresh)
		kernel_run.set_arg(3, mobj_img_shape_input)
		kernel_run.set_arg(4, mobj_img_shape_output)
				
		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		#worksize = (int(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )
				
		# ENQUEUEING KERNEL EXECUTION
		#cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.ipl.shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClNdBinToGray(self, img_input, img_output):

		if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinToGray: Error: this function supports only OpenCL data as buffer and img_input isn't.")
			exit()
		if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			print("vglClNdBinToGray: Error: this function supports only OpenCL data as buffer and img_output isn't.")
			exit()

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClNdBinToGray.cl", "vglClNdBinToGray")
		kernel_run = _program.vglClNdBinToGray

		mobj_img_shape_input = img_input.getVglShape().get_asVglClShape_buffer()
		mobj_img_shape_output = img_output.getVglShape().get_asVglClShape_buffer()

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, mobj_img_shape_input)
		kernel_run.set_arg(3, mobj_img_shape_output)
				
		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		#worksize = (np.int32(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )
				
		# ENQUEUEING KERNEL EXECUTION
		#cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.ipl.shape, None)
		
		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

"""
	HERE FOLLOWS THE KERNEL CALLS
"""
if __name__ == "__main__":
	
	wrp = cl2py_BIN_ND()

	# INPUT IMAGE
	img_input = vl.VglImage("bin.pgm", vl.VGL_IMAGE_2D_IMAGE(), None, vl.IMAGE_ND_ARRAY())
	vl.vglLoadImage(img_input)
	vl.vglClUpload(img_input)

	img_input2 = vl.VglImage("bin2.pgm", vl.VGL_IMAGE_2D_IMAGE(), None, vl.IMAGE_ND_ARRAY())
	vl.vglLoadImage(img_input2)
	vl.vglClUpload(img_input2)

	# OUTPUT IMAGE
	img_output = vl.create_blank_image_as(img_input)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
	vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())

	img_out_aux = vl.get_similar_oclPtr_object(img_input)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_out_aux, img_output.get_oclPtr())

	# TRANSFORMANDO IMAGENS EM IMAGENS BINARIAS
	wrp.vglClNdBinThreshold(img_input, img_output, 100)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())

	# STRUCTURANT ELEMENT
	window = vl.VglStrEl()
	window.constructorFromTypeNdim(vl.VGL_STREL_CROSS(), 2)
	
	# INPUT IMAGE
	img_input = vl.VglImage("bin.pgm", vl.VGL_IMAGE_2D_IMAGE(), None, vl.IMAGE_ND_ARRAY())
	vl.vglLoadImage(img_input)
	vl.vglClUpload(img_input)
	wrp.vglClNdBinThreshold(img_input, img_output, 100)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())

	wrp.vglClNdBinDilate(img_input, img_output, window)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())
	wrp.vglClNdBinToGray(img_input, img_output)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("bin-vglClNdBinDilate.png", img_output)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
	
	img_input = vl.VglImage("bin.pgm", vl.VGL_IMAGE_2D_IMAGE(), None, vl.IMAGE_ND_ARRAY())
	vl.vglLoadImage(img_input)
	vl.vglClUpload(img_input)
	wrp.vglClNdBinThreshold(img_input, img_output, 100)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())

	wrp.vglClNdBinDilatePack(img_input, img_output, window)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())
	wrp.vglClNdBinToGray(img_input, img_output)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("bin-vglClNdBinDilatePack.png", img_output)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )

	img_input = vl.VglImage("bin.pgm", vl.VGL_IMAGE_2D_IMAGE(), None, vl.IMAGE_ND_ARRAY())
	vl.vglLoadImage(img_input)
	vl.vglClUpload(img_input)
	wrp.vglClNdBinThreshold(img_input, img_output, 100)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())

	wrp.vglClNdBinErode(img_input, img_output, window)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())
	wrp.vglClNdBinToGray(img_input, img_output)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("bin-vglClNdBinErode.png", img_output)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )

	img_input = vl.VglImage("bin.pgm", vl.VGL_IMAGE_2D_IMAGE(), None, vl.IMAGE_ND_ARRAY())
	vl.vglLoadImage(img_input)
	vl.vglClUpload(img_input)
	wrp.vglClNdBinThreshold(img_input, img_output, 100)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())

	wrp.vglClNdBinErodePack(img_input, img_output, window)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())
	wrp.vglClNdBinToGray(img_input, img_output)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("bin-vglClNdBinErodePack.png", img_output)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )

	img_input = vl.VglImage("bin.pgm", vl.VGL_IMAGE_2D_IMAGE(), None, vl.IMAGE_ND_ARRAY())
	vl.vglLoadImage(img_input)
	vl.vglClUpload(img_input)
	wrp.vglClNdBinThreshold(img_input, img_output, 100)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())

	wrp.vglClNdBinMax(img_input, img_input2, img_output)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())
	wrp.vglClNdBinToGray(img_input, img_output)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("bin-vglClNdBinMax.png", img_output)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )

	img_input = vl.VglImage("bin.pgm", vl.VGL_IMAGE_2D_IMAGE(), None, vl.IMAGE_ND_ARRAY())
	vl.vglLoadImage(img_input)
	vl.vglClUpload(img_input)
	wrp.vglClNdBinThreshold(img_input, img_output, 100)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())

	wrp.vglClNdBinMin(img_input, img_input2, img_output)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())
	wrp.vglClNdBinToGray(img_input, img_output)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("bin-vglClNdBinMin.png", img_output)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )

	img_input = vl.VglImage("bin.pgm", vl.VGL_IMAGE_2D_IMAGE(), None, vl.IMAGE_ND_ARRAY())
	vl.vglLoadImage(img_input)
	vl.vglClUpload(img_input)
	wrp.vglClNdBinThreshold(img_input, img_output, 100)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())

	wrp.vglClNdBinNot(img_input, img_output)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())
	wrp.vglClNdBinToGray(img_input, img_output)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("bin-vglClNdBinNot.png", img_output)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )

	"""
	img_input = vl.VglImage("bin.pgm", vl.VGL_IMAGE_2D_IMAGE(), None, vl.IMAGE_ND_ARRAY())
	vl.vglLoadImage(img_input)
	vl.vglClUpload(img_input)
	wrp.vglClNdBinThreshold(img_input, img_output, 100)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())

	wrp.vglClNdBinRoi(img_output, np.uint32((25, 25)), np.uint32((75, 75)))
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr())
	wrp.vglClNdBinToGray(img_input, img_output)
	vl.vglSaveImage("bin-vglClNdBinRoi.pgm", img_output)
	"""

	#wrp.vglClNdBinToGray(img_input, img_output)
	#vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	#vl.vglSaveImage("bin-vglClNdBinToGray.pgm", img_output)

	wrp = None
	img_input = None
	img_output = None
	window = None
