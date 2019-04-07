# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

# TO INFER TYPE TO THE VARIABLE
from typing import Union

#TO WORK WITH MAIN
import numpy as np
import sys

class cl2py_CL:
	def __init__(self, cl_ctx=None):
		# PYTHON-EXCLUSIVE VARIABLES
		self.cl_ctx: Union[None, vl.opencl_context] = cl_ctx

		# COMMON VARIABLES. self.ocl IS EQUIVALENT TO cl.
		self.ocl: Union[None, vl.VglClContext] = None

		# SE O CONTEXTO OPENCL NÃO FOR DEFINIDO
		# ELE INSTANCIADO E DEFINIDO
		if( self.cl_ctx is None ):
			vl.vglClInit()
			self.ocl = vl.get_ocl()
			self.cl_ctx = vl.get_ocl_context()
		else:
			self.ocl = cl_ctx.get_vglClContext_attributes()

	def vglClBinThreshold(self, img_input, img_output, thresh):

		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		if( not isinstance(thresh, np.float32) ):
			print("vglClBinThreshold: Warning: thresh not np.float32! Trying to convert...")
			try:
				thresh = np.float32(thresh)
			except Exception as e:
				print("vglClBinThreshold: Error!! Impossible to convert thresh as a np.float32 object.")
				print(str(e))
				exit()
		
		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClBinThreshold.cl", "vglClBinThreshold")
		kernel_run = _program.vglClBinThreshold

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, thresh)
			
		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		worksize = (int(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )

		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		#cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClBinConway(self, img_input, img_output):
		
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClBinConway.cl", "vglClBinConway")
		kernel_run = _program.vglClBinConway

		mobj_img_shape = img_input.getVglShape().get_asVglClShape_buffer()

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, mobj_img_shape)

		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		worksize = (int(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )

		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		#cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClBinNot(self, img_input, img_output):
		
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClBinNot.cl", "vglClBinNot")
		kernel_run = _program.vglClBinNot

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())

		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		worksize = (int(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )

		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		#cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClBinToGray(self, img_input, img_output):
		
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		_program = self.cl_ctx.get_compiled_kernel("../CL_BIN/vglClBinToGray.cl", "vglClBinToGray")
		kernel_run = _program.vglClBinToGray

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())

		_worksize_0 = img_input.getWidthIn()
		if( img_input.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_input.getWidthStep()
		if( img_output.depth == vl.IPL_DEPTH_1U() ):
			_worksize_0 = img_output.getWidthStep()
		
		worksize = (int(_worksize_0), img_input.getHeigthIn(), img_input.getNFrames() )

		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, worksize, None)
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())


"""
	HERE FOLLOWS THE KERNEL CALLS
"""

def salvando2d(img, name):
	# SAVING IMAGE img
	ext = name.split(".")
	ext.reverse()
	print(img.ipl.shape)

	#vl.vglClDownload(img)
	vl.vglCheckContext(img, vl.VGL_RAM_CONTEXT())

	if( ext.pop(0).lower() == 'jpg' ):
		if( img.getVglShape().getNChannels() == 4 ):
			vl.rgba_to_rgb(img)
	
	vl.vglSaveImage(name, img)

if __name__ == "__main__":
	
	wrp = cl2py_CL()

	"""
		CL.IMAGE OBJECTS
	"""

	img_input = vl.VglImage("bin.pgm", vl.VGL_IMAGE_2D_IMAGE())
	vl.vglLoadImage(img_input)
	#if( img_input.getVglShape().getNChannels() == 3 ):
	#	vl.rgb_to_rgba(img_input)
	
	vl.vglClUpload(img_input)
	
	img_input2 = vl.VglImage("bin.pgm", vl.VGL_IMAGE_2D_IMAGE())
	vl.vglLoadImage(img_input2)
	#if( img_input2.getVglShape().getNChannels() == 3 ):
	#	vl.rgb_to_rgba(img_input2)

	#img_input_3d = vl.VglImage("3d.tif", vl.VGL_IMAGE_3D_IMAGE())
	#vl.vglLoadImage(img_input_3d)
	#vl.vglClUpload(img_input_3d)

	#img_input2_3d = vl.VglImage("3d-2.tif", vl.VGL_IMAGE_3D_IMAGE())
	#vl.vglLoadImage(img_input2_3d)

	img_output = vl.create_blank_image_as(img_input)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
	vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())

	#img_output_3d = vl.create_blank_image_as(img_input_3d)
	#img_output_3d.set_oclPtr( vl.get_similar_oclPtr_object(img_input_3d) )
	#vl.vglAddContext(img_output_3d, vl.VGL_CL_CONTEXT())

	convolution_window_2d = np.ones((5,5), np.float32) * (1/25)
	convolution_window_3d = np.ones((5,5,5), np.float32) * (1/125)

	morph_window_2d = np.ones((3,3), np.uint8) * 255
	morph_window_2d[0,0] = 0 
	morph_window_2d[0,2] = 0
	morph_window_2d[2,0] = 0
	morph_window_2d[2,2] = 0

	morph_window_3d = np.zeros((3,3,3), np.uint8)
	morph_window_3d[0,1,1] = 255
	morph_window_3d[1,0,1] = 255
	morph_window_3d[1,1,0] = 255
	morph_window_3d[1,1,1] = 255
	morph_window_3d[1,1,2] = 255
	morph_window_3d[1,2,1] = 255
	morph_window_3d[2,1,1] = 255

	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_input, vl.VGL_RAM_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())	

	wrp.vglClBinThreshold(img_input, img_output, np.float32(.5))
	
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr(), dest_origin=(0,0,0), src_origin=(0,0,0), region=img_input.get_oclPtr().shape)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )

	wrp.vglClBinNot(img_input, img_output)
	cl.enqueue_copy(wrp.cl_ctx.queue, img_input.get_oclPtr(), img_output.get_oclPtr(), dest_origin=(0,0,0), src_origin=(0,0,0), region=img_input.get_oclPtr().shape)

	wrp.vglClBinToGray(img_input, img_output)
	salvando2d(img_output, "bin-vglClBinNot.pgm")

	wrp = None
	
	img_input = None
	img_input_3d = None

	img_input2 = None
	img_input2_3d = None
	
	img_output = None
	img_output_3d = None
	
	convolution_window_2d = None
	convolution_window_3d = None
	
	morph_window_2d = None
	morph_window_3d = None