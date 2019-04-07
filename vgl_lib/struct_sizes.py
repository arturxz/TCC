import pyopencl as cl
import numpy as np

import vgl_lib as vl

"""
	struct_sizes GETS THE WAY OPENCL DEVICE
	ORGANIZES STRUCT DATA AND STORES IT
	INTO THE struct_sizes_host .
"""
class struct_sizes:
	def __init__(self):
		print("-> struct_sizes: Starting")
		
		# INSTANTIATING THE VARIABLE THAT WILL STORE THE DATA
		self.struct_sizes_host = np.zeros(11, np.uint32)

		# GETTING EXISTING DATA
		self.ocl_ctx = vl.get_ocl_context()
		
		# IF THERE'S NO CONTEXT, CREATE ONE AND GETS IT
		if( self.ocl_ctx is None ):
			vl.vglClInit()
			self.ocl_ctx = vl.get_ocl_context()
		
		# COMPILING KERNEL THAT WILL RETURN DATA ORGANIZATION
		self._program = self.ocl_ctx.get_compiled_kernel("vgl_lib/get_struct_sizes.cl", "get_struct_sizes")
		self.kernel_run = self._program.get_struct_sizes

		self.execute()
		print("<- struct_sizes: Ending\n")

	def execute(self):

		# GETTING DEVICE POINTER AND COPYING DATA TO IT
		self.struct_sizes_device = cl.Buffer( self.ocl_ctx.ctx, cl.mem_flags.READ_ONLY, self.struct_sizes_host.nbytes )

		# EXECUTING KERNEL WITH THE IMAGES
		print("struct_sizes: Executing kernel")
		self.kernel_run.set_arg( 0, self.struct_sizes_device )

		#EXECUTING KERNEL AND COPYING DATA BACK TO RAM
		cl.enqueue_nd_range_kernel(self.ocl_ctx.queue, self.kernel_run, self.struct_sizes_host.shape, None)
		cl.enqueue_copy(self.ocl_ctx.queue, self.struct_sizes_host, self.struct_sizes_device, is_blocking=True)

	def get_struct_sizes(self):
		return self.struct_sizes_host