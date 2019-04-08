import pyopencl as cl
import numpy as np
import sys
from vglImage import *

class vgl:
	# THE vgl CONSTRUCTOR CREATES A NEW CONTEXT
	# AND INITIATES THE QUEUE, ADDING QUE CONTEXT TO IT.
	def __init__(self):
		print("Starting OpenCL")
		self.platform = cl.get_platforms()[0]
		self.devs = self.platform.get_devices()
		self.device = self.devs[0]
		self.ctx = cl.Context([self.device])
		#self.ctx = cl.create_some_context()
		self.queue = cl.CommandQueue(self.ctx)
		self.builded = False

	# THIS FUNCTION WILL LOAD THE KERNEL FILE
	# AND BUILD IT IF NECESSARY.
	def loadCL(self, filepath):
		print("Loading OpenCL Kernel")
		self.kernel_file = open(filepath, "r")

		if ((self.builded == False)):
			self.pgr = cl.Program(self.ctx, self.kernel_file.read())
			self.pgr.build()
			self.kernel_file.close()
			self.builded = True
		else:
			print("Kernel already builded. Going to next step...")

	def loadImage(self, imgpath):
		print("Opening image to be processed")
		
		self.vglimage = VglImage(imgpath)
		if( self.vglimage.getVglShape().getNChannels() == 3 ):
			self.vglimage.rgb_to_rgba()

		self.vglimage.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage.get_similar_device_image_object(self.ctx, self.queue)
	
	def execute(self, outputpath):
		# EXECUTING KERNEL WITH THE IMAGES
		print("Executing kernel")
		self.pgr.vglClInvert(self.queue, 
							 self.img_out_cl.shape, 
							 None, 
							 self.vglimage.get_device_image(), 
							 self.img_out_cl).wait()
		
		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)
		if( self.vglimage.getVglShape().getNChannels() == 4 ):
			self.vglimage.rgba_to_rgb()
		self.vglimage.img_save(outputpath)


CLPath = "../../CL/vglClInvert.cl"
inPath = sys.argv[1]
ouPath = sys.argv[2] 

process = vgl()
process.loadCL(CLPath)
process.loadImage(inPath)
process.execute(ouPath)