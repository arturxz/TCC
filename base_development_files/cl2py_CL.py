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

	def vglClBlurSq3(self, img_input, img_output):
		print("# Running vglClBlurSq3")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL/vglClBlurSq3.cl", "vglClBlurSq3")
		kernel_run = _program.vglClBlurSq3

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	
	def vglClConvolution(self, img_input, img_output, convolution_window, window_size_x, window_size_y):
		print("# Running vglClConvolution")		
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		# TRANSFORMAR EM BUFFER
		try:
			cl_convolution_window = cl.Buffer(self.ocl.context, cl.mem_flags.READ_ONLY, convolution_window.nbytes)
			cl.enqueue_copy(self.ocl.commandQueue, cl_convolution_window, convolution_window.tobytes(), is_blocking=True)
			convolution_window = cl_convolution_window
		except Exception as e:
			print("vglClConvolution: Error!! Impossible to convert convolution_window to cl.Buffer object.")
			print(str(e))
			exit()
		
		if( not isinstance(window_size_x, np.uint32) ):
			print("vglClConvolution: Warning: window_size_x not np.uint32! Trying to convert...")
			try:
				window_size_x = np.uint32(window_size_x)
			except Exception as e:
				print("vglClConvolution: Error!! Impossible to convert window_size_x as a np.uint32 object.")
				print(str(e))
				exit()
		
		if( not isinstance(window_size_y, np.uint32) ):
			print("vglClConvolution: Warning: window_size_y not np.uint32! Trying to convert...")
			try:
				window_size_y = np.uint32(window_size_y)
			except Exception as e:
				print("vglClConvolution: Error!! Impossible to convert window_size_y as a np.uint32 object.")
				print(str(e))
				exit()

		_program = self.cl_ctx.get_compiled_kernel("../CL/vglClConvolution.cl", "vglClConvolution")
		kernel_run = _program.vglClConvolution

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, convolution_window)
		kernel_run.set_arg(3, window_size_x)
		kernel_run.set_arg(4, window_size_y)

		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClCopy(self, img_input, img_output):
		print("# Running vglClCopy")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		_program = self.cl_ctx.get_compiled_kernel("../CL/vglClCopy.cl", "vglClCopy")
		kernel_run = _program.vglClCopy

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClDilate(self, img_input, img_output, convolution_window, window_size_x, window_size_y):
		print("# Running vglClDilate")		
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		# TRANSFORMAR EM BUFFER
		try:
			cl_convolution_window = cl.Buffer(self.ocl.context, cl.mem_flags.READ_ONLY, convolution_window.nbytes)
			cl.enqueue_copy(self.ocl.commandQueue, cl_convolution_window, convolution_window.tobytes(), is_blocking=True)
			convolution_window = cl_convolution_window
		except Exception as e:
			print("vglClDilate: Error!! Impossible to convert convolution_window to cl.Buffer object.")
			print(str(e))
			exit()

		if( not isinstance(window_size_x, np.uint32) ):
			print("vglClDilate: Warning: window_size_x not np.uint32! Trying to convert...")
			try:
				window_size_x = np.uint32(window_size_x)
			except Exception as e:
				print("vglClDilate: Error!! Impossible to convert window_size_x as a np.uint32 object.")
				print(str(e))
				exit()
		if( not isinstance(window_size_y, np.uint32) ):
			print("vglClDilate: Warning: window_size_y not np.uint32! Trying to convert...")
			try:
				window_size_y = np.uint32(window_size_y)
			except Exception as e:
				print("vglClDilate: Error!! Impossible to convert window_size_y as a np.uint32 object.")
				print(str(e))
				exit()

		_program = self.cl_ctx.get_compiled_kernel("../CL/vglClDilate.cl", "vglClDilate")
		kernel_run = _program.vglClDilate

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, convolution_window)
		kernel_run.set_arg(3, window_size_x)
		kernel_run.set_arg(4, window_size_y)

		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClErode(self, img_input, img_output, convolution_window, window_size_x, window_size_y):
		print("# Running vglClErode")		
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		# TRANSFORMAR EM BUFFER
		try:
			cl_convolution_window = cl.Buffer(self.ocl.context, cl.mem_flags.READ_ONLY, convolution_window.nbytes)
			cl.enqueue_copy(self.ocl.commandQueue, cl_convolution_window, convolution_window.tobytes(), is_blocking=True)
			convolution_window = cl_convolution_window
		except Exception as e:
			print("vglClErode: Error!! Impossible to convert convolution_window to cl.Buffer object.")
			print(str(e))
			exit()

		if( not isinstance(window_size_x, np.uint32) ):
			print("vglClErode: Warning: window_size_x not np.uint32! Trying to convert...")
			try:
				window_size_x = np.uint32(window_size_x)
			except Exception as e:
				print("vglClErode: Error!! Impossible to convert window_size_x as a np.uint32 object.")
				print(str(e))
				exit()
		
		if( not isinstance(window_size_y, np.uint32) ):
			print("vglClErode: Warning: window_size_y not np.uint32! Trying to convert...")
			try:
				window_size_y = np.uint32(window_size_y)
			except Exception as e:
				print("vglClErode: Error!! Impossible to convert window_size_y as a np.uint32 object.")
				print(str(e))
				exit()

		_program = self.cl_ctx.get_compiled_kernel("../CL/vglClErode.cl", "vglClErode")
		kernel_run = _program.vglClErode

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, convolution_window)
		kernel_run.set_arg(3, window_size_x)
		kernel_run.set_arg(4, window_size_y)

		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClInvert(self, img_input, img_output):
		print("# Running vglClInvert")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		_program = self.cl_ctx.get_compiled_kernel("../CL/vglClInvert.cl", "vglClInvert")
		kernel_run = _program.vglClInvert

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClMax(self, img_input, img_input2, img_output):
		print("# Running vglClMax")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		_program = self.cl_ctx.get_compiled_kernel("../CL/vglClMax.cl", "vglClMax")
		kernel_run = _program.vglClMax

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_input2.get_oclPtr())
		kernel_run.set_arg(2, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClMin(self, img_input, img_input2, img_output):
		print("# Running vglClMin")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL/vglClMin.cl", "vglClMin")
		kernel_run = _program.vglClMin

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_input2.get_oclPtr())
		kernel_run.set_arg(2, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClSub(self, img_input, img_input2, img_output):
		print("# Running vglClSub")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL/vglClSub.cl", "vglClSub")
		kernel_run = _program.vglClSub

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_input2.get_oclPtr())
		kernel_run.set_arg(2, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClSum(self, img_input, img_input2, img_output):
		print("# Running vglClSum")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL/vglClSum.cl", "vglClSum")
		kernel_run = _program.vglClSum

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_input2.get_oclPtr())
		kernel_run.set_arg(2, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	
	def vglClSwapRgb(self, img_input, img_output):
		print("# Running vglClSwapRgb")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL/vglClSwapRgb.cl", "vglClSwapRgb")
		kernel_run = _program.vglClSwapRgb

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClThreshold(self, img_input, img_output, thresh, top=1.0):
		print("# Running vglClThreshold")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		if( not isinstance(thresh, np.float32) ):
			print("vglClThreshold: Warning: thresh not np.float32! Trying to convert...")
			try:
				thresh = np.float32(thresh)
			except Exception as e:
				print("vglClThreshold: Error!! Impossible to convert thresh as a np.float32 object.")
				print(str(e))
				exit()
		
		if( not isinstance(top, np.float32) ):
			print("vglClThreshold: Warning: top not np.float32! Trying to convert...")
			try:
				top = np.float32(top)
			except Exception as e:
				print("vglClThreshold: Error!! Impossible to convert top as a np.float32 object.")
				print(str(e))
				exit()
		
		_program = self.cl_ctx.get_compiled_kernel("../CL/vglClThreshold.cl", "vglClThreshold")
		kernel_run = _program.vglClThreshold

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, thresh)
		kernel_run.set_arg(3, top)
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dBlurSq3(self, img_input, img_output):
		print("# Running vglCl3dBlurSq3")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL/vglCl3dBlurSq3.cl", "vglCl3dBlurSq3")
		kernel_run = _program.vglCl3dBlurSq3

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	
	def vglCl3dConvolution(self, img_input, img_output, convolution_window, window_size_x, window_size_y, window_size_z):
		print("# Running vglCl3dConvolution")		
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		# TRANSFORMAR EM BUFFER
		try:
			cl_convolution_window = cl.Buffer(self.ocl.context, cl.mem_flags.READ_ONLY, convolution_window.nbytes)
			cl.enqueue_copy(self.ocl.commandQueue, cl_convolution_window, convolution_window.tobytes(), is_blocking=True)
			convolution_window = cl_convolution_window
		except Exception as e:
			print("vglCl3dConvolution: Error!! Impossible to convert convolution_window to cl.Buffer object.")
			print(str(e))
			exit()

		if( not isinstance(window_size_x, np.uint32) ):
			print("vglCl3dConvolution: Warning: window_size_x not np.uint32! Trying to convert...")
			try:
				window_size_x = np.uint32(window_size_x)
			except Exception as e:
				print("vglCl3dConvolution: Error!! Impossible to convert window_size_x as a np.uint32 object.")
				print(str(e))
				exit()
		
		if( not isinstance(window_size_y, np.uint32) ):
			print("vglCl3dConvolution: Warning: window_size_y not np.uint32! Trying to convert...")
			try:
				window_size_y = np.uint32(window_size_y)
			except Exception as e:
				print("vglCl3dConvolution: Error!! Impossible to convert window_size_y as a np.uint32 object.")
				print(str(e))
				exit()
		
		if( not isinstance(window_size_z, np.uint32) ):
			print("vglCl3dConvolution: Warning: window_size_z not np.uint32! Trying to convert...")
			try:
				window_size_z = np.uint32(window_size_z)
			except Exception as e:
				print("vglCl3dConvolution: Error!! Impossible to convert window_size_z as a np.uint32 object.")
				print(str(e))
				exit()
		
		_program = self.cl_ctx.get_compiled_kernel("../CL/vglCl3dConvolution.cl", "vglCl3dConvolution")
		kernel_run = _program.vglCl3dConvolution

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, convolution_window)
		kernel_run.set_arg(3, window_size_x)
		kernel_run.set_arg(4, window_size_y)
		kernel_run.set_arg(5, window_size_z)

		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dCopy(self, img_input, img_output):
		print("# Running vglCl3dCopy")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		_program = self.cl_ctx.get_compiled_kernel("../CL/vglCl3dCopy.cl", "vglCl3dCopy")
		kernel_run = _program.vglCl3dCopy

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dDilate(self, img_input, img_output, convolution_window, window_size_x, window_size_y, window_size_z):
		print("# Running vglCl3dDilate")		
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		# TRANSFORMAR EM BUFFER
		try:
			cl_convolution_window = cl.Buffer(self.ocl.context, cl.mem_flags.READ_ONLY, convolution_window.nbytes)
			cl.enqueue_copy(self.ocl.commandQueue, cl_convolution_window, convolution_window.tobytes(), is_blocking=True)
			convolution_window = cl_convolution_window
		except Exception as e:
			print("vglCl3dDilate: Error!! Impossible to convert convolution_window to cl.Buffer object.")
			print(str(e))
			exit()

		if( not isinstance(window_size_x, np.uint32) ):
			print("vglCl3dDilate: Warning: window_size_x not np.uint32! Trying to convert...")
			try:
				window_size_x = np.uint32(window_size_x)
			except Exception as e:
				print("vglCl3dDilate: Error!! Impossible to convert window_size_x as a np.uint32 object.")
				print(str(e))
				exit()
		
		if( not isinstance(window_size_y, np.uint32) ):
			print("vglCl3dDilate: Warning: window_size_y not np.uint32! Trying to convert...")
			try:
				window_size_y = np.uint32(window_size_y)
			except Exception as e:
				print("vglCl3dDilate: Error!! Impossible to convert window_size_y as a np.uint32 object.")
				print(str(e))
				exit()
		
		if( not isinstance(window_size_z, np.uint32) ):
			print("vglCl3dDilate: Warning: window_size_z not np.uint32! Trying to convert...")
			try:
				window_size_z = np.uint32(window_size_z)
			except Exception as e:
				print("vglCl3dDilate: Error!! Impossible to convert window_size_z as a np.uint32 object.")
				print(str(e))
				exit()
		
		_program = self.cl_ctx.get_compiled_kernel("../CL/vglCl3dDilate.cl", "vglCl3dDilate")
		kernel_run = _program.vglCl3dDilate

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, convolution_window)
		kernel_run.set_arg(3, window_size_x)
		kernel_run.set_arg(4, window_size_y)
		kernel_run.set_arg(5, window_size_z)

		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dErode(self, img_input, img_output, convolution_window, window_size_x, window_size_y, window_size_z):
		print("# Running vglCl3dErode")		
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
		# TRANSFORMAR EM BUFFER
		try:
			cl_convolution_window = cl.Buffer(self.ocl.context, cl.mem_flags.READ_ONLY, convolution_window.nbytes)
			cl.enqueue_copy(self.ocl.commandQueue, cl_convolution_window, convolution_window.tobytes(), is_blocking=True)
			convolution_window = cl_convolution_window
		except Exception as e:
			print("vglCl3dErode: Error!! Impossible to convert convolution_window to cl.Buffer object.")
			print(str(e))
			exit()

		if( not isinstance(window_size_x, np.uint32) ):
			print("vglCl3dErode: Warning: window_size_x not np.uint32! Trying to convert...")
			try:
				window_size_x = np.uint32(window_size_x)
			except Exception as e:
				print("vglCl3dErode: Error!! Impossible to convert window_size_x as a np.uint32 object.")
				print(str(e))
				exit()
		
		if( not isinstance(window_size_y, np.uint32) ):
			print("vglCl3dErode: Warning: window_size_y not np.uint32! Trying to convert...")
			try:
				window_size_y = np.uint32(window_size_y)
			except Exception as e:
				print("vglCl3dErode: Error!! Impossible to convert window_size_y as a np.uint32 object.")
				print(str(e))
				exit()
		if( not isinstance(window_size_z, np.uint32) ):
			print("vglCl3dErode: Warning: window_size_z not np.uint32! Trying to convert...")
			try:
				window_size_z = np.uint32(window_size_z)
			except Exception as e:
				print("vglCl3dErode: Error!! Impossible to convert window_size_z as a np.uint32 object.")
				print(str(e))
				exit()
		
		_program = self.cl_ctx.get_compiled_kernel("../CL/vglCl3dErode.cl", "vglCl3dErode")
		kernel_run = _program.vglCl3dErode

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, convolution_window)
		kernel_run.set_arg(3, window_size_x)
		kernel_run.set_arg(4, window_size_y)
		kernel_run.set_arg(5, window_size_z)

		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dNot(self, img_input, img_output):
		print("# Running vglCl3dNot")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL/vglCl3dNot.cl", "vglCl3dNot")
		kernel_run = _program.vglCl3dNot

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dMax(self, img_input, img_input2, img_output):
		print("# Running vglCl3dMax")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
	
		_program = self.cl_ctx.get_compiled_kernel("../CL/vglCl3dMax.cl", "vglCl3dMax")
		kernel_run = _program.vglCl3dMax

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_input2.get_oclPtr())
		kernel_run.set_arg(2, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dMin(self, img_input, img_input2, img_output):
		print("# Running vglCl3dMin")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL/vglCl3dMin.cl", "vglCl3dMin")
		kernel_run = _program.vglCl3dMin

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_input2.get_oclPtr())
		kernel_run.set_arg(2, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
		
		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dSub(self, img_input, img_input2, img_output):
		print("# Running vglCl3dSub")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL/vglCl3dSub.cl", "vglCl3dSub")
		kernel_run = _program.vglCl3dSub

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_input2.get_oclPtr())
		kernel_run.set_arg(2, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dSum(self, img_input, img_input2, img_output):
		print("# Running vglCl3dSum")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		_program = self.cl_ctx.get_compiled_kernel("../CL/vglCl3dSum.cl", "vglCl3dSum")
		kernel_run = _program.vglCl3dSum

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_input2.get_oclPtr())
		kernel_run.set_arg(2, img_output.get_oclPtr())
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	
	def vglCl3dThreshold(self, img_input, img_output, thresh, top=1.0):
		print("# Running vglCl3dThreshold")
		vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
		vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

		if( not isinstance(thresh, np.float32) ):
			print("vglCl3dThreshold: Warning: thresh not np.float32! Trying to convert...")
			try:
				thresh = np.float32(thresh)
			except Exception as e:
				print("vglCl3dThreshold: Error!! Impossible to convert thresh as a np.float32 object.")
				print(str(e))
				exit()
		
		if( not isinstance(top, np.float32) ):
			print("vglCl3dThreshold: Warning: top not np.float32! Trying to convert...")
			try:
				top = np.float32(top)
			except Exception as e:
				print("vglCl3dThreshold: Error!! Impossible to convert top as a np.float32 object.")
				print(str(e))
				exit()
		
		_program = self.cl_ctx.get_compiled_kernel("../CL/vglCl3dThreshold.cl", "vglCl3dThreshold")
		kernel_run = _program.vglCl3dThreshold

		kernel_run.set_arg(0, img_input.get_oclPtr())
		kernel_run.set_arg(1, img_output.get_oclPtr())
		kernel_run.set_arg(2, thresh)
		kernel_run.set_arg(3, top)
			
		cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

		vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())



"""
	HERE FOLLOWS THE KERNEL CALLS
"""

def salvando2d(img, name):
	# SAVING IMAGE img
	ext = name.split(".")
	ext.reverse()

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

	img_input = vl.VglImage("yamamoto.jpg", None, vl.VGL_IMAGE_2D_IMAGE())
	vl.vglLoadImage(img_input)
	if( img_input.getVglShape().getNChannels() == 3 ):
		vl.rgb_to_rgba(img_input)
	
	vl.vglClUpload(img_input)
	
	img_input2 = vl.VglImage("yamamoto-vglClThreshold.jpg", None, vl.VGL_IMAGE_2D_IMAGE())
	vl.vglLoadImage(img_input2)
	if( img_input2.getVglShape().getNChannels() == 3 ):
		vl.rgb_to_rgba(img_input2)

	img_input_3d = vl.VglImage("3d.tif", None, vl.VGL_IMAGE_3D_IMAGE())
	vl.vglLoadImage(img_input_3d)
	vl.vglClUpload(img_input_3d)

	img_input2_3d = vl.VglImage("3d-2.tif", None, vl.VGL_IMAGE_3D_IMAGE())
	vl.vglLoadImage(img_input2_3d)

	img_output = vl.create_blank_image_as(img_input)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
	vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())

	img_output_3d = vl.create_blank_image_as(img_input_3d)
	img_output_3d.set_oclPtr( vl.get_similar_oclPtr_object(img_input_3d) )
	vl.vglAddContext(img_output_3d, vl.VGL_CL_CONTEXT())

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

	wrp.vglClBlurSq3(img_input, img_output)
	salvando2d(img_output, "yamamoto-vglClBlurSq3.jpg")
	vl.rgb_to_rgba(img_output)
	
	wrp.vglClConvolution(img_input, img_output, convolution_window_2d, np.uint32(5), np.uint32(5))
	salvando2d(img_output, "yamamoto-vglClConvolution.jpg")
	vl.rgb_to_rgba(img_output)

	wrp.vglClCopy(img_input, img_output)
	salvando2d(img_output, "yamamoto-vglClCopy.jpg")
	vl.rgb_to_rgba(img_output)

	wrp.vglClDilate(img_input, img_output, morph_window_2d, np.uint32(3), np.uint32(3))
	salvando2d(img_output, "yamamoto-vglClDilate.jpg")
	vl.rgb_to_rgba(img_output)

	wrp.vglClErode(img_input, img_output, morph_window_2d, np.uint32(3), np.uint32(3))
	salvando2d(img_output, "yamamoto-vglClErode.jpg")
	vl.rgb_to_rgba(img_output)

	wrp.vglClInvert(img_input, img_output)
	salvando2d(img_output, "yamamoto-vglClInvert.jpg")
	vl.rgb_to_rgba(img_output)

	wrp.vglClMax(img_input, img_input2, img_output)
	salvando2d(img_output, "yamamoto-vglClMax.jpg")
	vl.rgb_to_rgba(img_output)

	wrp.vglClMin(img_input, img_input2, img_output)
	salvando2d(img_output, "yamamoto-vglClMin.jpg")
	vl.rgb_to_rgba(img_output)

	wrp.vglClSub(img_input, img_input2, img_output)
	salvando2d(img_output, "yamamoto-vglClSub.jpg")
	vl.rgb_to_rgba(img_output)

	wrp.vglClSum(img_input, img_input2, img_output)
	salvando2d(img_output, "yamamoto-vglClSum.jpg")
	vl.rgb_to_rgba(img_output)

	wrp.vglClSwapRgb(img_input, img_output)
	salvando2d(img_output, "yamamoto-vglClSwapRgb.jpg")
	vl.rgb_to_rgba(img_output)

	wrp.vglClThreshold(img_input, img_output, np.float32(0.5), np.float32(0.9))
	salvando2d(img_output, "yamamoto-vglClThreshold.jpg")
	vl.rgb_to_rgba(img_output)
	
	wrp.vglCl3dBlurSq3(img_input_3d, img_output_3d)
	salvando2d(img_output_3d, "3d-vglCl3dBlurSq3.tif")

	wrp.vglCl3dConvolution(img_input_3d, img_output_3d, convolution_window_3d, np.uint32(5), np.uint32(5), np.uint32(5))
	salvando2d(img_output_3d, "3d-vglCl3dConvolution.tif")

	wrp.vglCl3dCopy(img_input_3d, img_output_3d)
	salvando2d(img_output_3d, "3d-vglCl3dCopy.tif")

	wrp.vglCl3dDilate(img_input_3d, img_output_3d, morph_window_3d, np.uint32(3), np.uint32(3), np.uint32(3))
	salvando2d(img_output_3d, "3d-vglCl3dDilate.tif")

	wrp.vglCl3dErode(img_input_3d, img_output_3d, morph_window_3d, np.uint32(3), np.uint32(3), np.uint32(3))
	salvando2d(img_output_3d, "3d-vglCl3dErode.tif")

	wrp.vglCl3dNot(img_input_3d, img_output_3d)
	salvando2d(img_output_3d, "3d-vglCl3dNot.tif")

	wrp.vglCl3dMax(img_input_3d, img_input2_3d, img_output_3d)
	salvando2d(img_output_3d, "3d-vglCl3dMax.tif")

	wrp.vglCl3dMin(img_input_3d, img_input2_3d, img_output_3d)
	salvando2d(img_output_3d, "3d-vglCl3dMin.tif")

	wrp.vglCl3dSub(img_input_3d, img_input2_3d, img_output_3d)
	salvando2d(img_output_3d, "3d-vglCl3dSub.tif")

	wrp.vglCl3dSum(img_input_3d, img_input2_3d, img_output_3d)
	salvando2d(img_output_3d, "3d-vglCl3dSum.tif")

	wrp.vglCl3dThreshold(img_input_3d, img_output_3d, np.float32(0.4), np.float32(.8))
	salvando2d(img_output_3d, "3d-vglCl3dThreshold.tif")
	
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