import pyopencl as cl
import vgl_lib as vl
import sys, glob, os

"""
	THIS CLASS IS EQUIVALENT TO VglClContext
	STRUCT FOUNDED ON vglClImage.h

	HERE, IT IS USED JUST TO PASS THE PLATFORM ID,
	THE DEVICE ID, THE CONTEXT AND THE QUEUE OF THE
	DEVICE IN THE WAY IT IS FOUNDED ON C/C++ VERSION 
"""
class VglClContext:
	def __init__(self, pl, dv, cn, cq):
		self.platformId = pl
		self.deviceId = dv
		self.context = cn
		self.commandQueue = cq

class opencl_context:
	"""
		THIS CLASS MANAGES THE PyOpenCL INICIAL INSTANCIATION AND
		THE SYSTEM'S DEVICES AND ITS PROPERTIES (LIKE CONTEXT AND QUEUE).
		IT ALSO LOAD THE HEADERS AND CONSTANTS NEEDED TO COMPILE THE KERNELS.
	"""

	def __init__(self):
		print("opencl_context: Instanciating Context and Queue...")
		self.platform = cl.get_platforms()[0]
		self.devs = self.platform.get_devices()
		self.device = self.devs[0]
		self.ctx = cl.Context([self.device])
		self.queue = cl.CommandQueue(self.ctx)

		# PROGRAM VARIABLE. STORES ALL COMPILED KERNELS
		self.programs = []

	def is_kernel_compiled(self, method_name):
		for program in self.programs:
			if( method_name in program.kernel_names ):
				return program
		return None

	"""
		THIS METHOD VERIFIES IF PATH TO KERNEL FILE EXISTS.
			IF EXISTS, IT VERIFIES IF THE KERNEL IS ALREADY COMPILED.
			IF IS NOT COMPILED, THE KERNEL IS BUILDED.
			IF THE KERNEL IS ALREADY COMPILED, IT DOES NOTHING.
	"""
	def get_compiled_kernel(self, filepath, kernelname):
		print("get_compiled_kernel: Loading OpenCL Kernel")
		kernel_file = None

		try:
			kernel_file = open(filepath, "r")
		except FileNotFoundError as fnf:
			print("get_compiled_kernel: Error: Kernel File not found. Filepath:", filepath)    
			print(str(fnf))
			exit()
		except Exception as e:
			print("get_compiled_kernel: Error: A unexpected exception was thrown while trying to open kernel file. Filepath:", filepath)
			print(str(e))
			exit()

		program = self.is_kernel_compiled(kernelname)
		
		if( program is None ):
			print("get_compiled_kernel: Building Kernel.")
			self.load_headers(filepath)
			program = cl.Program(self.ctx, kernel_file.read())
			self.programs.append( program.build(options=self.get_build_options()) )
			
		kernel_file.close()
		return self.is_kernel_compiled(kernelname)
		
	"""
		Making the vglClContext variables to retrocompatibility, where
			self.platformId = self.platform.int_ptr
			self.deviceId = self.device.int_ptr
			self.context = self.ctx
			self.commandQueue = self.queue
	"""
	def get_vglClContext_attributes(self):
		return VglClContext(self.platform.int_ptr, self.device.int_ptr, self.ctx, self.queue)
	
	def load_headers(self, filepath):
		print("Loading Headers")
		self.kernel_file = open(filepath, "r")
		buildDir = self.getDir(filepath)

		self.build_options = "-I "+buildDir

		# READING THE HEADER FILES BEFORE COMPILING THE KERNEL
		while( buildDir ):
			for file in glob.glob(buildDir+"/*.h"):
				#print(file)
				self.pgr = cl.Program(self.ctx, open(file, "r"))
			
			buildDir = self.getDir(buildDir)

	def getDir(self, filePath):
		size = len(filePath)-1
		bar = -1
		for i in range(0, size):
			if(filePath[i] == '/'):
				bar = i
				i = -1
		return filePath[:bar+1]
	
	# GETTERS
	def get_queue(self):
		return self.queue
	
	def get_context(self):
		return self.ctx
	
	def get_build_options(self):
		return self.build_options