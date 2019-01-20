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
		self._program: Union[None, cl.Program] = None
		self.ocl: Union[None, vl.VglClContext] = None

		if( self.cl_ctx is None ):
			vl.vglClInit()
			self.ocl = vl.get_ocl()
			self.cl_ctx = vl.get_ocl_context()
		else:
			self.ocl = cl_ctx.get_vglClContext_attributes()

	def load_kernel(self, filepath, kernelname):
		print("Loading OpenCL Kernel")
		kernel_file = open(filepath, "r")

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

	def vglClBlurSq3(self, img_input, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglClBlurSq3.cl", "vglClBlurSq3")
			kernel_run = self._program.vglClBlurSq3

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	
	def vglClConvolution(self, img_input, img_output, convolution_window, window_size_x, window_size_y):
		
		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( not isinstance(convolution_window, cl.Buffer) ):
			print("vglClConvolution: Error: convolution_window is not cl.Buffer object.")
			exit()
		elif( not isinstance(window_size_x, np.uint32) ):
			print("vglClConvolution: Warning: window_size_x not np.uint32! Trying to convert...")
			try:
				window_size_x = np.uint32(window_size_x)
			except Exception as e:
				print("vglClConvolution: Error!! Impossible to convert window_size_x as a np.uint32 object.")
				print(str(e))
				exit()
		elif( not isinstance(window_size_y, np.uint32) ):
			print("vglClConvolution: Warning: window_size_y not np.uint32! Trying to convert...")
			try:
				window_size_y = np.uint32(window_size_y)
			except Exception as e:
				print("vglClConvolution: Error!! Impossible to convert window_size_y as a np.uint32 object.")
				print(str(e))
				exit()
		else:
			self.load_kernel("../CL/vglClConvolution.cl", "vglClConvolution")
			kernel_run = self._program.vglClConvolution

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			kernel_run.set_arg(2, convolution_window)
			kernel_run.set_arg(3, window_size_x)
			kernel_run.set_arg(4, window_size_y)

			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClCopy(self, img_input, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglClCopy.cl", "vglClCopy")
			kernel_run = self._program.vglClCopy

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClDilate(self, img_input, img_output, convolution_window, window_size_x, window_size_y):
		
		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( not isinstance(convolution_window, cl.Buffer) ):
			print("vglClDilate: Error: convolution_window is not cl.Buffer object.")
			exit()
		elif( not isinstance(window_size_x, np.uint32) ):
			print("vglClDilate: Warning: window_size_x not np.uint32! Trying to convert...")
			try:
				window_size_x = np.uint32(window_size_x)
			except Exception as e:
				print("vglClDilate: Error!! Impossible to convert window_size_x as a np.uint32 object.")
				print(str(e))
				exit()
		elif( not isinstance(window_size_y, np.uint32) ):
			print("vglClDilate: Warning: window_size_y not np.uint32! Trying to convert...")
			try:
				window_size_y = np.uint32(window_size_y)
			except Exception as e:
				print("vglClDilate: Error!! Impossible to convert window_size_y as a np.uint32 object.")
				print(str(e))
				exit()
		else:
			self.load_kernel("../CL/vglClDilate.cl", "vglClDilate")
			kernel_run = self._program.vglClDilate

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			kernel_run.set_arg(2, convolution_window)
			kernel_run.set_arg(3, window_size_x)
			kernel_run.set_arg(4, window_size_y)

			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClErode(self, img_input, img_output, convolution_window, window_size_x, window_size_y):
		
		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( not isinstance(convolution_window, cl.Buffer) ):
			print("vglClErode: Error: convolution_window is not cl.Buffer object.")
			exit()
		elif( not isinstance(window_size_x, np.uint32) ):
			print("vglClErode: Warning: window_size_x not np.uint32! Trying to convert...")
			try:
				window_size_x = np.uint32(window_size_x)
			except Exception as e:
				print("vglClErode: Error!! Impossible to convert window_size_x as a np.uint32 object.")
				print(str(e))
				exit()
		elif( not isinstance(window_size_y, np.uint32) ):
			print("vglClErode: Warning: window_size_y not np.uint32! Trying to convert...")
			try:
				window_size_y = np.uint32(window_size_y)
			except Exception as e:
				print("vglClErode: Error!! Impossible to convert window_size_y as a np.uint32 object.")
				print(str(e))
				exit()
		else:
			self.load_kernel("../CL/vglClErode.cl", "vglClErode")
			kernel_run = self._program.vglClErode

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			kernel_run.set_arg(2, convolution_window)
			kernel_run.set_arg(3, window_size_x)
			kernel_run.set_arg(4, window_size_y)

			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClInvert(self, img_input, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglClInvert.cl", "vglClInvert")
			kernel_run = self._program.vglClInvert

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClMax(self, img_input, img_input2, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglClMax.cl", "vglClMax")
			kernel_run = self._program.vglClMax

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_input2.get_oclPtr())
			kernel_run.set_arg(2, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClMin(self, img_input, img_input2, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglClMin.cl", "vglClMin")
			kernel_run = self._program.vglClMin

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_input2.get_oclPtr())
			kernel_run.set_arg(2, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClSub(self, img_input, img_input2, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglClSub.cl", "vglClSub")
			kernel_run = self._program.vglClSub

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_input2.get_oclPtr())
			kernel_run.set_arg(2, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClSum(self, img_input, img_input2, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglClSum.cl", "vglClSum")
			kernel_run = self._program.vglClSum

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_input2.get_oclPtr())
			kernel_run.set_arg(2, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	
	def vglClSwapRgb(self, img_input, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglClSwapRgb.cl", "vglClSwapRgb")
			kernel_run = self._program.vglClSwapRgb

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglClThreshold(self, img_input, img_output, thresh, top):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( not isinstance(thresh, np.float32) ):
			print("vglClThreshold: Warning: thresh not np.float32! Trying to convert...")
			try:
				thresh = np.float32(thresh)
			except Exception as e:
				print("vglClThreshold: Error!! Impossible to convert thresh as a np.float32 object.")
				print(str(e))
				exit()
		elif( not isinstance(top, np.float32) ):
			print("vglClThreshold: Warning: top not np.float32! Trying to convert...")
			try:
				top = np.float32(top)
			except Exception as e:
				print("vglClThreshold: Error!! Impossible to convert top as a np.float32 object.")
				print(str(e))
				exit()
		else:
			self.load_kernel("../CL/vglClThreshold.cl", "vglClThreshold")
			kernel_run = self._program.vglClThreshold

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			kernel_run.set_arg(2, thresh)
			kernel_run.set_arg(3, top)
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dBlurSq3(self, img_input, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglCl3dBlurSq3.cl", "vglCl3dBlurSq3")
			kernel_run = self._program.vglCl3dBlurSq3

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	
	def vglCl3dConvolution(self, img_input, img_output, convolution_window, window_size_x, window_size_y, window_size_z):
		
		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( not isinstance(convolution_window, cl.Buffer) ):
			print("vglCl3dConvolution: Error: convolution_window is not cl.Buffer object.")
			exit()
		elif( not isinstance(window_size_x, np.uint32) ):
			print("vglCl3dConvolution: Warning: window_size_x not np.uint32! Trying to convert...")
			try:
				window_size_x = np.uint32(window_size_x)
			except Exception as e:
				print("vglCl3dConvolution: Error!! Impossible to convert window_size_x as a np.uint32 object.")
				print(str(e))
				exit()
		elif( not isinstance(window_size_y, np.uint32) ):
			print("vglCl3dConvolution: Warning: window_size_y not np.uint32! Trying to convert...")
			try:
				window_size_y = np.uint32(window_size_y)
			except Exception as e:
				print("vglCl3dConvolution: Error!! Impossible to convert window_size_y as a np.uint32 object.")
				print(str(e))
				exit()
		elif( not isinstance(window_size_z, np.uint32) ):
			print("vglCl3dConvolution: Warning: window_size_z not np.uint32! Trying to convert...")
			try:
				window_size_z = np.uint32(window_size_z)
			except Exception as e:
				print("vglCl3dConvolution: Error!! Impossible to convert window_size_z as a np.uint32 object.")
				print(str(e))
				exit()
		else:
			self.load_kernel("../CL/vglCl3dConvolution.cl", "vglCl3dConvolution")
			kernel_run = self._program.vglCl3dConvolution

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			kernel_run.set_arg(2, convolution_window)
			kernel_run.set_arg(3, window_size_x)
			kernel_run.set_arg(4, window_size_y)
			kernel_run.set_arg(5, window_size_z)

			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dCopy(self, img_input, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglCl3dCopy.cl", "vglCl3dCopy")
			kernel_run = self._program.vglCl3dCopy

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dDilate(self, img_input, img_output, convolution_window, window_size_x, window_size_y, window_size_z):
		
		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( not isinstance(convolution_window, cl.Buffer) ):
			print("vglCl3dDilate: Error: convolution_window is not cl.Buffer object.")
			exit()
		elif( not isinstance(window_size_x, np.uint32) ):
			print("vglCl3dDilate: Warning: window_size_x not np.uint32! Trying to convert...")
			try:
				window_size_x = np.uint32(window_size_x)
			except Exception as e:
				print("vglCl3dDilate: Error!! Impossible to convert window_size_x as a np.uint32 object.")
				print(str(e))
				exit()
		elif( not isinstance(window_size_y, np.uint32) ):
			print("vglCl3dDilate: Warning: window_size_y not np.uint32! Trying to convert...")
			try:
				window_size_y = np.uint32(window_size_y)
			except Exception as e:
				print("vglCl3dDilate: Error!! Impossible to convert window_size_y as a np.uint32 object.")
				print(str(e))
				exit()
		elif( not isinstance(window_size_z, np.uint32) ):
			print("vglCl3dDilate: Warning: window_size_z not np.uint32! Trying to convert...")
			try:
				window_size_z = np.uint32(window_size_z)
			except Exception as e:
				print("vglCl3dDilate: Error!! Impossible to convert window_size_z as a np.uint32 object.")
				print(str(e))
				exit()
		else:
			self.load_kernel("../CL/vglCl3dDilate.cl", "vglCl3dDilate")
			kernel_run = self._program.vglCl3dDilate

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			kernel_run.set_arg(2, convolution_window)
			kernel_run.set_arg(3, window_size_x)
			kernel_run.set_arg(4, window_size_y)
			kernel_run.set_arg(5, window_size_z)

			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dErode(self, img_input, img_output, convolution_window, window_size_x, window_size_y, window_size_z):
		
		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( not isinstance(convolution_window, cl.Buffer) ):
			print("vglCl3dErode: Error: convolution_window is not cl.Buffer object.")
			exit()
		elif( not isinstance(window_size_x, np.uint32) ):
			print("vglCl3dErode: Warning: window_size_x not np.uint32! Trying to convert...")
			try:
				window_size_x = np.uint32(window_size_x)
			except Exception as e:
				print("vglCl3dErode: Error!! Impossible to convert window_size_x as a np.uint32 object.")
				print(str(e))
				exit()
		elif( not isinstance(window_size_y, np.uint32) ):
			print("vglCl3dErode: Warning: window_size_y not np.uint32! Trying to convert...")
			try:
				window_size_y = np.uint32(window_size_y)
			except Exception as e:
				print("vglCl3dErode: Error!! Impossible to convert window_size_y as a np.uint32 object.")
				print(str(e))
				exit()
		elif( not isinstance(window_size_z, np.uint32) ):
			print("vglCl3dErode: Warning: window_size_z not np.uint32! Trying to convert...")
			try:
				window_size_z = np.uint32(window_size_z)
			except Exception as e:
				print("vglCl3dErode: Error!! Impossible to convert window_size_z as a np.uint32 object.")
				print(str(e))
				exit()
		else:
			self.load_kernel("../CL/vglCl3dErode.cl", "vglCl3dErode")
			kernel_run = self._program.vglCl3dErode

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			kernel_run.set_arg(2, convolution_window)
			kernel_run.set_arg(3, window_size_x)
			kernel_run.set_arg(4, window_size_y)
			kernel_run.set_arg(5, window_size_z)

			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dNot(self, img_input, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglCl3dNot.cl", "vglCl3dNot")
			kernel_run = self._program.vglCl3dNot

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dMax(self, img_input, img_input2, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglCl3dMax.cl", "vglCl3dMax")
			kernel_run = self._program.vglCl3dMax

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_input2.get_oclPtr())
			kernel_run.set_arg(2, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dMin(self, img_input, img_input2, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglCl3dMin.cl", "vglCl3dMin")
			kernel_run = self._program.vglCl3dMin

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_input2.get_oclPtr())
			kernel_run.set_arg(2, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dSub(self, img_input, img_input2, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglCl3dSub.cl", "vglCl3dSub")
			kernel_run = self._program.vglCl3dSub

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_input2.get_oclPtr())
			kernel_run.set_arg(2, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

	def vglCl3dSum(self, img_input, img_input2, img_output):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		else:
			self.load_kernel("../CL/vglCl3dSum.cl", "vglCl3dSum")
			kernel_run = self._program.vglCl3dSum

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_input2.get_oclPtr())
			kernel_run.set_arg(2, img_output.get_oclPtr())
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	
	def vglCl3dThreshold(self, img_input, img_output, thresh, top):

		if( vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT()) == vl.VGL_ERROR() ):
			exit()
		elif( not isinstance(thresh, np.float32) ):
			print("vglCl3dThreshold: Warning: thresh not np.float32! Trying to convert...")
			try:
				thresh = np.float32(thresh)
			except Exception as e:
				print("vglCl3dThreshold: Error!! Impossible to convert thresh as a np.float32 object.")
				print(str(e))
				exit()
		elif( not isinstance(top, np.float32) ):
			print("vglCl3dThreshold: Warning: top not np.float32! Trying to convert...")
			try:
				top = np.float32(top)
			except Exception as e:
				print("vglCl3dThreshold: Error!! Impossible to convert top as a np.float32 object.")
				print(str(e))
				exit()
		else:
			self.load_kernel("../CL/vglCl3dThreshold.cl", "vglCl3dThreshold")
			kernel_run = self._program.vglCl3dThreshold

			kernel_run.set_arg(0, img_input.get_oclPtr())
			kernel_run.set_arg(1, img_output.get_oclPtr())
			kernel_run.set_arg(2, thresh)
			kernel_run.set_arg(3, top)
			
			ev = cl.enqueue_nd_range_kernel(self.ocl.commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
			print(ev)

			vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())



"""
	HERE FOLLOWS THE KERNEL CALLS
"""
if __name__ == "__main__":
	
	wrp = cl2py_CL()

	"""
		CL.IMAGE OBJECTS
	"""
	img_input_morph_2d = vl.VglImage(sys.argv[1], vl.VGL_IMAGE_3D_IMAGE())
	vl.vglLoadImage(img_input_morph_2d)
	"""
	if( img_input_morph_2d.getVglShape().getNChannels() == 3 ):
		vl.rgb_to_rgba(img_input_morph_2d)
	"""
	
	vl.vglClUpload(img_input_morph_2d)

	img_output_morph_2d = vl.create_blank_image_as(img_input_morph_2d)
	img_output_morph_2d.set_oclPtr( vl.get_similar_oclPtr_object(img_input_morph_2d) )
	vl.vglAddContext(img_output_morph_2d, vl.VGL_CL_CONTEXT())

	img_input_2d = vl.VglImage("", vl.VGL_IMAGE_3D_IMAGE())
	#img_input_2d = vl.VglImage("yamamoto-dilate.jpg", vl.VGL_IMAGE_3D_IMAGE())
	vl.vglLoadImage(img_input_2d, sys.argv[1])
	"""
	if( img_input_2d.getVglShape().getNChannels() == 3 ):
		vl.rgb_to_rgba(img_input_2d)
	"""
	
	vl.vglClUpload(img_input_2d)

	img_input2_2d = vl.VglImage("3d-treshold.tif", vl.VGL_IMAGE_3D_IMAGE())
	vl.vglLoadImage(img_input2_2d)
	if( img_input2_2d.getVglShape().getNChannels() == 3 ):
		vl.rgb_to_rgba(img_input2_2d)

	img_output_2d = vl.create_blank_image_as(img_input_2d)
	img_output_2d.set_oclPtr( vl.get_similar_oclPtr_object(img_input_2d) )
	vl.vglAddContext(img_output_2d, vl.VGL_CL_CONTEXT())

	convolution_window_morph_2d = np.ones((3, 3, 3), np.float32)
	convolution_window_morph_2d[0,0,1] = np.float32(0)
	convolution_window_morph_2d[0,1,0] = np.float32(0)
	convolution_window_morph_2d[0,1,1] = np.float32(0)
	convolution_window_morph_2d[0,1,2] = np.float32(0)
	convolution_window_morph_2d[0,2,1] = np.float32(0)
	convolution_window_morph_2d[1,0,1] = np.float32(0)
	convolution_window_morph_2d[1,1,0] = np.float32(0)
	convolution_window_morph_2d[1,1,1] = np.float32(0)
	convolution_window_morph_2d[1,1,2] = np.float32(0)
	convolution_window_morph_2d[1,2,1] = np.float32(0)
	convolution_window_morph_2d[2,0,1] = np.float32(0)
	convolution_window_morph_2d[2,1,0] = np.float32(0)
	convolution_window_morph_2d[2,1,1] = np.float32(0)
	convolution_window_morph_2d[2,1,2] = np.float32(0)
	convolution_window_morph_2d[2,2,1] = np.float32(0)

	convolution_window_morph_2d = cl.Buffer(wrp.ocl.context , cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=convolution_window_morph_2d)

	convolution_window_2d = np.ones((5, 5, 5), np.float32) * (1/125)
	convolution_window_cl = cl.Buffer(wrp.ocl.context , cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=convolution_window_2d)
	
	#wrp.vglClBlurSq3(img_input_2d, img_output_2d)
	#wrp.vglClConvolution(img_input_2d, img_output_2d, convolution_window_cl, np.uint32(5), np.uint32(5))
	#wrp.vglClCopy(img_input_2d, img_output_2d)
	#wrp.vglClDilate(img_input_morph_2d, img_output_morph_2d, convolution_window_morph_2d, np.uint32(3), np.uint32(3))
	#wrp.vglClErode(img_input_morph_2d, img_output_morph_2d, convolution_window_morph_2d, np.uint32(3), np.uint32(3))
	#wrp.vglClInvert(img_input_2d, img_output_2d)
	#wrp.vglClMax(img_input_2d, img_input2_2d, img_output_2d)
	#wrp.vglClMin(img_input_2d, img_input2_2d, img_output_2d)
	#wrp.vglClSub(img_input_2d, img_input2_2d, img_output_2d)
	#wrp.vglClSum(img_input_2d, img_input2_2d, img_output_2d)
	#wrp.vglClSwapRgb(img_input_2d, img_output_2d)
	#wrp.vglClThreshold(img_input_2d, img_output_2d, np.float32(0.5), np.float32(0.9))

	#wrp.vglCl3dBlurSq3(img_input_2d, img_output_2d)
	#wrp.vglCl3dConvolution(img_input_2d, img_output_2d, convolution_window_cl, np.uint32(5), np.uint32(5), np.uint32(5))
	#wrp.vglCl3dCopy(img_input_2d, img_output_2d)
	#wrp.vglCl3dDilate(img_input_morph_2d, img_output_morph_2d, convolution_window_morph_2d, np.uint32(3), np.uint32(3), np.uint32(3))
	#wrp.vglCl3dErode(img_input_morph_2d, img_output_morph_2d, convolution_window_morph_2d, np.uint32(3), np.uint32(3), np.uint32(3))
	#wrp.vglCl3dNot(img_input_2d, img_output_2d)
	#wrp.vglCl3dMax(img_input_2d, img_input2_2d, img_output_2d)
	#wrp.vglCl3dMin(img_input_2d, img_input2_2d, img_output_2d)
	#wrp.vglCl3dSub(img_input_2d, img_input2_2d, img_output_2d)
	#wrp.vglCl3dSum(img_input_2d, img_input2_2d, img_output_2d)
	#wrp.vglCl3dThreshold(img_input_2d, img_output_2d, np.float32(0.4), np.float32(.8))

	#vl.vglClDownload(img_output_morph_2d)
	vl.vglClDownload(img_output_2d)
	

	# SAVING IMAGE img_output
	ext = sys.argv[2].split(".")
	ext.reverse()

	vl.vglCheckContext(img_output_2d, vl.VGL_RAM_CONTEXT())

	"""
	if( ext.pop(0).lower() == 'jpg' ):
		if( img_output_2d.getVglShape().getNChannels() == 4 ):
			vl.rgba_to_rgb(img_output_2d)
	"""
	#vl.vglSaveImage(sys.argv[2], img_output_morph_2d)
	vl.vglSaveImage(sys.argv[2], img_output_2d)


"""
class Wrapper:
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
		self.vglstrel_buffer = cl.Buffer(self.ctx, mf.READ_ONLY, vgl_strel_obj.nbytes)
		self.vglshape_buffer = cl.Buffer(self.ctx, mf.READ_ONLY, vgl_shape_obj.nbytes)
				
		# COPYING DATA FROM HOST TO DEVICE
		cl.enqueue_copy(self.queue, self.vglstrel_buffer, vgl_strel_obj.tobytes(), is_blocking=True)
		cl.enqueue_copy(self.queue, self.vglshape_buffer, vgl_shape_obj.tobytes(), is_blocking=True)

	def copy_into_byte_array(self, value, byte_array, offset):
		for iterator, byte in enumerate( value.tobytes() ):
			byte_array[iterator+offset] = byte
"""