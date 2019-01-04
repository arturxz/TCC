import pyopencl as cl
import vgl_lib as vl
import sys, glob, os

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

		
		self.build_options = ""
		self.build_options = self.build_options + "-I "+buildDir
		"""
		self.build_options = self.build_options + " -D VGL_SHAPE_NCHANNELS={0}".format(vl.VGL_SHAPE_NCHANNELS())
		self.build_options = self.build_options + " -D VGL_SHAPE_WIDTH={0}".format(vl.VGL_SHAPE_WIDTH())
		self.build_options = self.build_options + " -D VGL_SHAPE_HEIGHT={0}".format(vl.VGL_SHAPE_HEIGHT())
		self.build_options = self.build_options + " -D VGL_SHAPE_LENGTH={0}".format(vl.VGL_SHAPE_LENGTH())
		self.build_options = self.build_options + " -D VGL_MAX_DIM={0}".format(vl.VGL_MAX_DIM())
		self.build_options = self.build_options + " -D VGL_ARR_SHAPE_SIZE={0}".format(vl.VGL_ARR_SHAPE_SIZE())
		self.build_options = self.build_options + " -D VGL_ARR_CLSTREL_SIZE={0}".format(vl.VGL_ARR_CLSTREL_SIZE())
		self.build_options = self.build_options + " -D VGL_STREL_CUBE={0}".format(vl.VGL_STREL_CUBE())
		self.build_options = self.build_options + " -D VGL_STREL_CROSS={0}".format(vl.VGL_STREL_CROSS())
		self.build_options = self.build_options + " -D VGL_STREL_GAUSS={0}".format(vl.VGL_STREL_GAUSS())
		self.build_options = self.build_options + " -D VGL_STREL_MEAN={0}".format(vl.VGL_STREL_MEAN())
		"""

		#print("Build Options:\n", self.build_options)

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