# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

#TO WORK WITH MAIN
import numpy as np

def vglClBlurSq3(img_input, img_output):
	print("# Running vglClBlurSq3")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglClBlurSq3.cl", "vglClBlurSq3")
	kernel_run = _program.vglClBlurSq3

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	
def vglClConvolution(img_input, img_output, convolution_window, window_size_x, window_size_y):
	print("# Running vglClConvolution")		
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
	# TRANSFORMAR EM BUFFER
	try:
		cl_convolution_window = cl.Buffer(vl.get_ocl().context, cl.mem_flags.READ_ONLY, convolution_window.nbytes)
		cl.enqueue_copy(vl.get_ocl().commandQueue, cl_convolution_window, convolution_window.tobytes(), is_blocking=True)
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

	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglClConvolution.cl", "vglClConvolution")
	kernel_run = _program.vglClConvolution

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
	kernel_run.set_arg(2, convolution_window)
	kernel_run.set_arg(3, window_size_x)
	kernel_run.set_arg(4, window_size_y)

	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglClCopy(img_input, img_output):
	print("# Running vglClCopy")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglClCopy.cl", "vglClCopy")
	kernel_run = _program.vglClCopy

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglClDilate(img_input, img_output, convolution_window, window_size_x, window_size_y):
	print("# Running vglClDilate")		
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
	# TRANSFORMAR EM BUFFER
	try:
		cl_convolution_window = cl.Buffer(vl.get_ocl().context, cl.mem_flags.READ_ONLY, convolution_window.nbytes)
		cl.enqueue_copy(vl.get_ocl().commandQueue, cl_convolution_window, convolution_window.tobytes(), is_blocking=True)
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

	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglClDilate.cl", "vglClDilate")
	kernel_run = _program.vglClDilate

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
	kernel_run.set_arg(2, convolution_window)
	kernel_run.set_arg(3, window_size_x)
	kernel_run.set_arg(4, window_size_y)

	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglClErode(img_input, img_output, convolution_window, window_size_x, window_size_y):
	print("# Running vglClErode")		
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
	# TRANSFORMAR EM BUFFER
	try:
		cl_convolution_window = cl.Buffer(vl.get_ocl().context, cl.mem_flags.READ_ONLY, convolution_window.nbytes)
		cl.enqueue_copy(vl.get_ocl().commandQueue, cl_convolution_window, convolution_window.tobytes(), is_blocking=True)
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

	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglClErode.cl", "vglClErode")
	kernel_run = _program.vglClErode

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
	kernel_run.set_arg(2, convolution_window)
	kernel_run.set_arg(3, window_size_x)
	kernel_run.set_arg(4, window_size_y)

	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglClInvert(img_input, img_output):
	print("# Running vglClInvert")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglClInvert.cl", "vglClInvert")
	kernel_run = _program.vglClInvert

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglClMax(img_input, img_input2, img_output):
	print("# Running vglClMax")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglClMax.cl", "vglClMax")
	kernel_run = _program.vglClMax

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_input2.get_oclPtr())
	kernel_run.set_arg(2, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglClMin(img_input, img_input2, img_output):
	print("# Running vglClMin")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglClMin.cl", "vglClMin")
	kernel_run = _program.vglClMin

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_input2.get_oclPtr())
	kernel_run.set_arg(2, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglClSub(img_input, img_input2, img_output):
	print("# Running vglClSub")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglClSub.cl", "vglClSub")
	kernel_run = _program.vglClSub

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_input2.get_oclPtr())
	kernel_run.set_arg(2, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglClSum(img_input, img_input2, img_output):
	print("# Running vglClSum")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglClSum.cl", "vglClSum")
	kernel_run = _program.vglClSum

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_input2.get_oclPtr())
	kernel_run.set_arg(2, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	
def vglClSwapRgb(img_input, img_output):
	print("# Running vglClSwapRgb")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglClSwapRgb.cl", "vglClSwapRgb")
	kernel_run = _program.vglClSwapRgb

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglClThreshold(img_input, img_output, thresh, top=1.0):
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
		
	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglClThreshold.cl", "vglClThreshold")
	kernel_run = _program.vglClThreshold

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
	kernel_run.set_arg(2, thresh)
	kernel_run.set_arg(3, top)
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglCl3dBlurSq3(img_input, img_output):
	print("# Running vglCl3dBlurSq3")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglCl3dBlurSq3.cl", "vglCl3dBlurSq3")
	kernel_run = _program.vglCl3dBlurSq3

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	
def vglCl3dConvolution(img_input, img_output, convolution_window, window_size_x, window_size_y, window_size_z):
	print("# Running vglCl3dConvolution")		
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
	# TRANSFORMAR EM BUFFER
	try:
		cl_convolution_window = cl.Buffer(vl.get_ocl().context, cl.mem_flags.READ_ONLY, convolution_window.nbytes)
		cl.enqueue_copy(vl.get_ocl().commandQueue, cl_convolution_window, convolution_window.tobytes(), is_blocking=True)
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
		
	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglCl3dConvolution.cl", "vglCl3dConvolution")
	kernel_run = _program.vglCl3dConvolution

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
	kernel_run.set_arg(2, convolution_window)
	kernel_run.set_arg(3, window_size_x)
	kernel_run.set_arg(4, window_size_y)
	kernel_run.set_arg(5, window_size_z)

	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglCl3dCopy(img_input, img_output):
	print("# Running vglCl3dCopy")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglCl3dCopy.cl", "vglCl3dCopy")
	kernel_run = _program.vglCl3dCopy

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglCl3dDilate(img_input, img_output, convolution_window, window_size_x, window_size_y, window_size_z):
	print("# Running vglCl3dDilate")		
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
	# TRANSFORMAR EM BUFFER
	try:
		cl_convolution_window = cl.Buffer(vl.get_ocl().context, cl.mem_flags.READ_ONLY, convolution_window.nbytes)
		cl.enqueue_copy(vl.get_ocl().commandQueue, cl_convolution_window, convolution_window.tobytes(), is_blocking=True)
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
		
	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglCl3dDilate.cl", "vglCl3dDilate")
	kernel_run = _program.vglCl3dDilate

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
	kernel_run.set_arg(2, convolution_window)
	kernel_run.set_arg(3, window_size_x)
	kernel_run.set_arg(4, window_size_y)
	kernel_run.set_arg(5, window_size_z)

	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglCl3dErode(img_input, img_output, convolution_window, window_size_x, window_size_y, window_size_z):
	print("# Running vglCl3dErode")		
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
		
	# TRANSFORMAR EM BUFFER
	try:
		cl_convolution_window = cl.Buffer(vl.get_ocl().context, cl.mem_flags.READ_ONLY, convolution_window.nbytes)
		cl.enqueue_copy(vl.get_ocl().commandQueue, cl_convolution_window, convolution_window.tobytes(), is_blocking=True)
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
		
	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglCl3dErode.cl", "vglCl3dErode")
	kernel_run = _program.vglCl3dErode

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
	kernel_run.set_arg(2, convolution_window)
	kernel_run.set_arg(3, window_size_x)
	kernel_run.set_arg(4, window_size_y)
	kernel_run.set_arg(5, window_size_z)

	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglCl3dNot(img_input, img_output):
	print("# Running vglCl3dNot")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglCl3dNot.cl", "vglCl3dNot")
	kernel_run = _program.vglCl3dNot

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglCl3dMax(img_input, img_input2, img_output):
	print("# Running vglCl3dMax")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
	
	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglCl3dMax.cl", "vglCl3dMax")
	kernel_run = _program.vglCl3dMax

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_input2.get_oclPtr())
	kernel_run.set_arg(2, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglCl3dMin(img_input, img_input2, img_output):
	print("# Running vglCl3dMin")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglCl3dMin.cl", "vglCl3dMin")
	kernel_run = _program.vglCl3dMin

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_input2.get_oclPtr())
	kernel_run.set_arg(2, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)
		
	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglCl3dSub(img_input, img_input2, img_output):
	print("# Running vglCl3dSub")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglCl3dSub.cl", "vglCl3dSub")
	kernel_run = _program.vglCl3dSub

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_input2.get_oclPtr())
	kernel_run.set_arg(2, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

def vglCl3dSum(img_input, img_input2, img_output):
	print("# Running vglCl3dSum")
	vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())
	vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglCl3dSum.cl", "vglCl3dSum")
	kernel_run = _program.vglCl3dSum

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_input2.get_oclPtr())
	kernel_run.set_arg(2, img_output.get_oclPtr())
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())
	
def vglCl3dThreshold(img_input, img_output, thresh, top=1.0):
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
		
	_program = vl.get_ocl_context().get_compiled_kernel("../CL/vglCl3dThreshold.cl", "vglCl3dThreshold")
	kernel_run = _program.vglCl3dThreshold

	kernel_run.set_arg(0, img_input.get_oclPtr())
	kernel_run.set_arg(1, img_output.get_oclPtr())
	kernel_run.set_arg(2, thresh)
	kernel_run.set_arg(3, top)
			
	cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, kernel_run, img_output.get_oclPtr().shape, None)

	vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())