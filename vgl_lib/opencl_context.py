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
		print("## Instanciating Context and Queue...")
		self.platform = cl.get_platforms()[0]
		self.devs = self.platform.get_devices()
		print("## Selecting first avaliable device on System...")
		self.device = self.devs[0]
		self.ctx = cl.Context([self.device])
		self.queue = cl.CommandQueue(self.ctx)
		
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