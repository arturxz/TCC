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

"""
	HERE FOLLOWS THE KERNEL CALLS
"""
if __name__ == "__main__":
	
	# vglClBlurSq3
	wrp = cl2py_CL()

	img_input_morph = vl.VglImage(sys.argv[1], vl.VGL_IMAGE_2D_IMAGE())
	vl.vglLoadImage(img_input_morph)
	if( img_input_morph.getVglShape().getNChannels() == 3 ):
		vl.rgb_to_rgba(img_input_morph)
	
	vl.vglClUpload(img_input_morph)

	img_output_morph = vl.create_blank_image_as(img_input_morph)
	img_output_morph.set_oclPtr( vl.get_similar_oclPtr_object(img_input_morph) )
	vl.vglAddContext(img_output_morph, vl.VGL_CL_CONTEXT())

	img_input = vl.VglImage("", vl.VGL_IMAGE_2D_IMAGE())
	vl.vglLoadImage(img_input, sys.argv[1])
	if( img_input.getVglShape().getNChannels() == 3 ):
		vl.rgb_to_rgba(img_input)
	
	vl.vglClUpload(img_input)

	img_output = vl.create_blank_image_as(img_input)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
	vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())

	convolution_window_morph = np.ones((3, 3), np.float32)
	convolution_window_morph[0,1] = np.float32(0)
	convolution_window_morph[1,0] = np.float32(0)
	convolution_window_morph[1,1] = np.float32(0)
	convolution_window_morph[1,2] = np.float32(0)
	convolution_window_morph[2,1] = np.float32(0)

	convolution_window_morph = cl.Buffer(wrp.ocl.context , cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=convolution_window_morph)

	convolution_window_2d = np.ones((5, 5), np.float32) * (1/25)
	convolution_window_cl = cl.Buffer(wrp.ocl.context , cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=convolution_window_2d)
	
	#wrp.vglClBlurSq3(img_input, img_output)
	#wrp.vglClConvolution(img_input, img_output, convolution_window_cl, np.uint32(5), np.uint32(5))
	#wrp.vglClCopy(img_input, img_output)
	#wrp.vglClDilate(img_input_morph, img_output_morph, convolution_window_morph, np.uint32(3), np.uint32(3))
	#wrp.vglClErode(img_input_morph, img_output_morph, convolution_window_morph, np.uint32(3), np.uint32(3))
	wrp.vglClCopy(img_input, img_output)
	
	#vl.vglClDownload(img_output_morph)
	vl.vglClDownload(img_output)
	

	# SAVING IMAGE img_output
	ext = sys.argv[2].split(".")
	ext.reverse()

	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())

	if( ext.pop(0).lower() == 'jpg' ):
		if( img_output.getVglShape().getNChannels() == 4 ):
			vl.rgba_to_rgb(img_output)

	#vl.vglSaveImage(sys.argv[2], img_output_morph)
	vl.vglSaveImage(sys.argv[2], img_output)


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