# IMAGE MANIPULATION LIBRARYS
from skimage import io
import numpy as np
import sys

# OPENCL LIBRARYS
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

class Wrapper:
	def __init__(self):
		self.ocl_ctx = vl.VglOclContext()
		self.ctx = self.ocl_ctx.get_context()
		self.queue = self.ocl_ctx.get_queue()
		self.builded = False
	
	def loadCL(self, filepath):
		print("Loading OpenCL Kernel")
		self.kernel_file = open(filepath, "r")

		if ((self.builded == False)):
			print("::Building Kernel")
			self.ocl_ctx.load_headers(filepath)
			self.pgr = cl.Program(self.ctx, self.kernel_file.read())
			self.pgr.build(options=self.ocl_ctx.get_build_options())
			self.builded = True
		else:
			print("::Kernel already builded. Going to next step...")

		self.kernel_file.close()
	
	"""
		HERE FOLLOWS THE IMAGE LOADING METHODS
	"""
	# LOADING INTO A 3D IMAGE OBJECT
	def loadImage3D(self, imgpath):
		print("Opening image to be processed")
		
		self.vglimage = vl.VglImage(imgpath, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage.get_similar_device_image_object(self.ctx, self.queue)
	
	# LOADING INTO A 2D IMAGE OBJECT
	def loadImage2D(self, imgpath):
		print("Opening image to be processed")
		
		self.vglimage = vl.VglImage(imgpath, vl.VGL_IMAGE_2D_IMAGE())
		if( self.vglimage.getVglShape().getNChannels() == 3 ):
			self.vglimage.rgb_to_rgba()
			
		self.vglimage.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage.get_similar_device_image_object(self.ctx, self.queue)
	
	# LOADING INTO A NDARRAY IMAGE
	def loadImageND(self, imgpath):
		print("Opening image to be processed")
		
		self.vglimage = vl.VglImage(imgpath)
		
		mf = cl.mem_flags
		self.vglimage.vglNdImageUpload(self.ctx, self.queue)
		self.img_out_cl = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.vglimage.get_host_image().nbytes)
	
	"""
		HERE FOLLOWS THE METHODS THAT STRUCTURE THE 
		VGLSHAPE AND VGLSTREL BUFFERS TO OPENCL.
	"""
	def makeStructures(self, strElType, strElDim):
		print("Making Structures")
		mf = cl.mem_flags

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
		HERE FOLLOWS THE IMAGE SAVING METHOD
	"""
	def saveImage(self, outputpath):
		self.vglimage.img_save(outputpath)

	"""
		HERE FOLLOWS THE KERNEL CALLS
	"""
	def vglCl3dBlurSq3(self, filepath, imgIn):
		self.loadCL(filepath)
		self.loadImage3D(imgIn)

		self.pgr.vglCl3dBlurSq3(self.queue,
								self.img_out_cl.shape, 
								None, 
								self.vglimage.get_device_image(),
								self.img_out_cl).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)

	def vglCl3dConvolution(self, filepath, imgIn, arr_window, window_x, window_y, window_z):
		self.loadCL(filepath)
		self.loadImage3D(imgIn)

		self.pgr.vglCl3dConvolution(self.queue,
									self.img_out_cl.shape, 
									None, 
									self.vglimage.get_device_image(), 
									self.img_out_cl,
									arr_window,
									np.uint32(window_x),
									np.uint32(window_y),
									np.uint32(window_z)).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)
	
	def vglCl3dCopy(self, filepath, imgIn):
		self.loadCL(filepath)
		self.loadImage3D(imgIn)

		self.pgr.vglCl3dCopy(self.queue, 
							 self.img_out_cl.shape, 
							 None, 
							 self.vglimage.get_device_image(),
							 self.img_out_cl).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)

	def vglCl3dDilate(self, filepath, imgIn, arr_window, window_x, window_y, window_z):
		self.loadCL(filepath)
		self.loadImage3D(imgIn)

		self.pgr.vglCl3dDilate(self.queue,
							   self.img_out_cl.shape, 
							   None, 
							   self.vglimage.get_device_image(), 
							   self.img_out_cl,
							   arr_window,
							   np.uint32(window_x),
							   np.uint32(window_y),
							   np.uint32(window_z)).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)

	def vglCl3dErode(self, filepath, imgIn, arr_window, window_x, window_y, window_z):
		self.loadCL(filepath)
		self.loadImage3D(imgIn)

		self.pgr.vglCl3dErode(self.queue,
							  self.img_out_cl.shape, 
							  None, 
							  self.vglimage.get_device_image(), 
							  self.img_out_cl,
							  arr_window,
							  np.uint32(window_x),
							  np.uint32(window_y),
							  np.uint32(window_z)).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)

	def vglCl3dMax(self, filepath, imgIn1, imgIn2):
		self.loadCL(filepath)
		
		self.vglimage1 = vl.VglImage(imgIn1, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage1.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage1.get_similar_device_image_object(self.ctx, self.queue)

		self.vglimage2 = vl.VglImage(imgIn2, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage2.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage2.get_similar_device_image_object(self.ctx, self.queue)

		self.pgr.vglCl3dMax(self.queue, 
							self.img_out_cl.shape, 
							None, 
							self.vglimage1.get_device_image(),
							self.vglimage2.get_device_image(),
							self.img_out_cl).wait()

		self.vglimage1.set_device_image(self.img_out_cl)
		self.vglimage1.sync(self.ctx, self.queue)
		self.vglimage2.set_device_image(self.img_out_cl)
		self.vglimage2.sync(self.ctx, self.queue)

	def vglCl3dMin(self, filepath, imgIn1, imgIn2):
		self.loadCL(filepath)
		
		self.vglimage1 = vl.VglImage(imgIn1, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage1.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage1.get_similar_device_image_object(self.ctx, self.queue)

		self.vglimage2 = vl.VglImage(imgIn2, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage2.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage2.get_similar_device_image_object(self.ctx, self.queue)

		self.pgr.vglCl3dMin(self.queue, 
							self.img_out_cl.shape, 
							None, 
							self.vglimage1.get_device_image(),
							self.vglimage2.get_device_image(),
							self.img_out_cl).wait()

		self.vglimage1.set_device_image(self.img_out_cl)
		self.vglimage1.sync(self.ctx, self.queue)
		self.vglimage2.set_device_image(self.img_out_cl)
		self.vglimage2.sync(self.ctx, self.queue)

	def vglCl3dNot(self, filepath, imgIn):
		self.loadCL(filepath)
		self.loadImage3D(imgIn)

		self.pgr.vglCl3dNot(self.queue, 
							self.img_out_cl.shape, 
							None, 
							self.vglimage.get_device_image(),
							self.img_out_cl).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)

	def vglCl3dSub(self, filepath, imgIn1, imgIn2):
		self.loadCL(filepath)
		
		self.vglimage1 = vl.VglImage(imgIn1, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage1.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage1.get_similar_device_image_object(self.ctx, self.queue)

		self.vglimage2 = vl.VglImage(imgIn2, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage2.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage2.get_similar_device_image_object(self.ctx, self.queue)

		self.pgr.vglCl3dSub(self.queue, 
							self.img_out_cl.shape, 
							None, 
							self.vglimage1.get_device_image(),
							self.vglimage2.get_device_image(),
							self.img_out_cl).wait()

		self.vglimage1.set_device_image(self.img_out_cl)
		self.vglimage1.sync(self.ctx, self.queue)
		self.vglimage2.set_device_image(self.img_out_cl)
		self.vglimage2.sync(self.ctx, self.queue)

	def vglCl3dSum(self, filepath, imgIn1, imgIn2):
		self.loadCL(filepath)
		
		self.vglimage1 = vl.VglImage(imgIn1, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage1.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage1.get_similar_device_image_object(self.ctx, self.queue)

		self.vglimage2 = vl.VglImage(imgIn2, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage2.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage2.get_similar_device_image_object(self.ctx, self.queue)

		self.pgr.vglCl3dSum(self.queue, 
							self.img_out_cl.shape, 
							None, 
							self.vglimage1.get_device_image(),
							self.vglimage2.get_device_image(),
							self.img_out_cl).wait()

		self.vglimage1.set_device_image(self.img_out_cl)
		self.vglimage1.sync(self.ctx, self.queue)
		self.vglimage2.set_device_image(self.img_out_cl)
		self.vglimage2.sync(self.ctx, self.queue)

	def vglCl3dThreshold(self, filepath, imgIn, thresh=0.425, top=1):
		self.loadCL(filepath)
		self.loadImage3D(imgIn)

		self.pgr.vglCl3dThreshold(self.queue, 
								  self.img_out_cl.shape, 
								  None, 
								  self.vglimage.get_device_image(),
								  self.img_out_cl,
								  np.float32(thresh),
								  np.float32(top)).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)

	def vglClBlurSq3(self, filepath, imgIn):
		self.loadCL(filepath)
		self.loadImage2D(imgIn)

		self.pgr.vglClBlurSq3(self.queue,
							  self.img_out_cl.shape,
							  None, 
							  self.vglimage.get_device_image(), 
							  self.img_out_cl).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)
		if( self.vglimage.getVglShape().getNChannels() == 4 ):
			self.vglimage.rgba_to_rgb()

	def vglClConvolution(self, filepath, imgIn, arr_window, window_x, window_y):
		self.loadCL(filepath)
		self.loadImage2D(imgIn)

		self.pgr.vglClConvolution(self.queue,
								  self.img_out_cl.shape, 
								  None, 
								  self.vglimage.get_device_image(), 
								  self.img_out_cl,
								  arr_window,
								  np.uint32(window_x),
								  np.uint32(window_y)).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)
		if( self.vglimage.getVglShape().getNChannels() == 4 ):
			self.vglimage.rgba_to_rgb()
	
	def vglClCopy(self, filepath, imgIn):
		self.loadCL(filepath)
		self.loadImage2D(imgIn)

		self.pgr.vglClCopy(self.queue, 
						   self.img_out_cl.shape, 
						   None, 
						   self.vglimage.get_device_image(),
						   self.img_out_cl).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)
		if( self.vglimage.getVglShape().getNChannels() == 4 ):
			self.vglimage.rgba_to_rgb()

	def vglClDilate(self, filepath, imgIn, arr_window, window_x, window_y):
		self.loadCL(filepath)
		self.loadImage2D(imgIn)

		self.pgr.vglClDilate(self.queue,
							 self.img_out_cl.shape, 
							 None, 
							 self.vglimage.get_device_image(), 
							 self.img_out_cl,
							 arr_window,
							 np.uint32(window_x),
							 np.uint32(window_y)).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)
		if( self.vglimage.getVglShape().getNChannels() == 4 ):
			self.vglimage.rgba_to_rgb()
			
	def vglClErode(self, filepath, imgIn, arr_window, window_x, window_y):
		self.loadCL(filepath)
		self.loadImage2D(imgIn)

		self.pgr.vglClErode(self.queue,
							self.img_out_cl.shape, 
							None, 
							self.vglimage.get_device_image(), 
							self.img_out_cl,
							arr_window,
							np.uint32(window_x),
							np.uint32(window_y)).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)
		if( self.vglimage.getVglShape().getNChannels() == 4 ):
			self.vglimage.rgba_to_rgb()
			
	def vglClMax(self, filepath, imgIn1, imgIn2):
		self.loadCL(filepath)
		
		self.vglimage1 = vl.VglImage(imgIn1, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage1.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage1.get_similar_device_image_object(self.ctx, self.queue)

		self.vglimage2 = vl.VglImage(imgIn2, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage2.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage2.get_similar_device_image_object(self.ctx, self.queue)

		self.pgr.vglClMax(self.queue, 
						  self.img_out_cl.shape, 
						  None, 
						  self.vglimage1.get_device_image(),
						  self.vglimage2.get_device_image(),
						  self.img_out_cl).wait()

		self.vglimage1.set_device_image(self.img_out_cl)
		self.vglimage1.sync(self.ctx, self.queue)
		if( self.vglimage1.getVglShape().getNChannels() == 4 ):
			self.vglimage1.rgba_to_rgb()
			
		self.vglimage2.set_device_image(self.img_out_cl)
		self.vglimage2.sync(self.ctx, self.queue)
		if( self.vglimage2.getVglShape().getNChannels() == 4 ):
			self.vglimage2.rgba_to_rgb()
			
	def vglClMin(self, filepath, imgIn1, imgIn2):
		self.loadCL(filepath)
		
		self.vglimage1 = vl.VglImage(imgIn1, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage1.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage1.get_similar_device_image_object(self.ctx, self.queue)

		self.vglimage2 = vl.VglImage(imgIn2, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage2.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage2.get_similar_device_image_object(self.ctx, self.queue)

		self.pgr.vglClMin(self.queue, 
						  self.img_out_cl.shape, 
						  None, 
						  self.vglimage1.get_device_image(),
						  self.vglimage2.get_device_image(),
						  self.img_out_cl).wait()

		self.vglimage1.set_device_image(self.img_out_cl)
		self.vglimage1.sync(self.ctx, self.queue)
		if( self.vglimage1.getVglShape().getNChannels() == 4 ):
			self.vglimage1.rgba_to_rgb()
			
		self.vglimage2.set_device_image(self.img_out_cl)
		self.vglimage2.sync(self.ctx, self.queue)
		if( self.vglimage2.getVglShape().getNChannels() == 4 ):
			self.vglimage2.rgba_to_rgb()

	def vglClInvert(self, filepath, imgIn):
		self.loadCL(filepath)
		self.loadImage2D(imgIn)

		self.pgr.vglClInvert(self.queue, 
							 self.img_out_cl.shape, 
							 None, 
							 self.vglimage.get_device_image(),
							 self.img_out_cl).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)
		if( self.vglimage.getVglShape().getNChannels() == 4 ):
			self.vglimage.rgba_to_rgb()

	def vglClSub(self, filepath, imgIn1, imgIn2):
		self.loadCL(filepath)
		
		self.vglimage1 = vl.VglImage(imgIn1, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage1.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage1.get_similar_device_image_object(self.ctx, self.queue)

		self.vglimage2 = vl.VglImage(imgIn2, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage2.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage2.get_similar_device_image_object(self.ctx, self.queue)

		self.pgr.vglClSub(self.queue, 
						  self.img_out_cl.shape, 
						  None, 
						  self.vglimage1.get_device_image(),
						  self.vglimage2.get_device_image(),
						  self.img_out_cl).wait()

		self.vglimage1.set_device_image(self.img_out_cl)
		self.vglimage1.sync(self.ctx, self.queue)
		if( self.vglimage1.getVglShape().getNChannels() == 4 ):
			self.vglimage1.rgba_to_rgb()
			
		self.vglimage2.set_device_image(self.img_out_cl)
		self.vglimage2.sync(self.ctx, self.queue)
		if( self.vglimage2.getVglShape().getNChannels() == 4 ):
			self.vglimage2.rgba_to_rgb()
			
	def vglClSum(self, filepath, imgIn1, imgIn2):
		self.loadCL(filepath)
		
		self.vglimage1 = vl.VglImage(imgIn1, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage1.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage1.get_similar_device_image_object(self.ctx, self.queue)

		self.vglimage2 = vl.VglImage(imgIn2, vl.VGL_IMAGE_3D_IMAGE())
		self.vglimage2.vglImageUpload(self.ctx, self.queue)
		self.img_out_cl = self.vglimage2.get_similar_device_image_object(self.ctx, self.queue)

		self.pgr.vglClSum(self.queue, 
						  self.img_out_cl.shape, 
						  None, 
						  self.vglimage1.get_device_image(),
						  self.vglimage2.get_device_image(),
						  self.img_out_cl).wait()

		self.vglimage1.set_device_image(self.img_out_cl)
		self.vglimage1.sync(self.ctx, self.queue)
		if( self.vglimage1.getVglShape().getNChannels() == 4 ):
			self.vglimage1.rgba_to_rgb()
			
		self.vglimage2.set_device_image(self.img_out_cl)
		self.vglimage2.sync(self.ctx, self.queue)
		if( self.vglimage2.getVglShape().getNChannels() == 4 ):
			self.vglimage2.rgba_to_rgb()
			
	def vglClThreshold(self, filepath, imgIn, thresh=0.425, top=1):
		self.loadCL(filepath)
		self.loadImage2D(imgIn)

		self.pgr.vglClThreshold(self.queue, 
								self.img_out_cl.shape, 
								None, 
								self.vglimage.get_device_image(),
								self.img_out_cl,
								np.float32(thresh),
								np.float32(top)).wait()

		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.sync(self.ctx, self.queue)
		if( self.vglimage.getVglShape().getNChannels() == 4 ):
			self.vglimage.rgba_to_rgb()

	def vglClNdConvolution(self, filepath, imgIn, strElType, strElDim):
		self.loadCL(filepath)
		self.loadImageND(imgIn)
		self.makeStructures(strElType, strElDim)

		self.pgr.vglClNdConvolution(self.queue,
									self.vglimage.get_host_image().shape,
									None, 
									self.vglimage.get_device_image(), 
									self.img_out_cl,
									self.vglshape_buffer,
									self.vglstrel_buffer).wait()
		
		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.vglNdImageDownload(self.ctx, self.queue)
	
	def vglClNdCopy(self, filepath, imgIn):
		self.loadCL(filepath)
		self.loadImageND(imgIn)

		self.pgr.vglClNdCopy(self.queue,
							self.vglimage.get_host_image().shape,
							None,
							self.vglimage.get_device_image(),
							self.img_out_cl).wait()
		
		self.vglimage.set_device_image(self.img_out_cl)
		self.vglimage.vglNdImageDownload(self.ctx, self.queue)

if __name__ == "__main__":

	wrp = Wrapper()
	"""
	##
	# PASTA CL
	##
	
	# vglCl3dBlurSq3
	wrp.vglCl3dBlurSq3("../CL/vglCl3dBlurSq3.cl", sys.argv[1])
	wrp.saveImage(sys.argv[2])
	
	# vglCl3dConvolution
	arr_window = np.ones((5,5,5), np.float32) * (1/125)
	arr_window_cl = cl.Buffer(wrp.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr_window)
	wrp.vglCl3dConvolution("../CL/vglCl3dConvolution.cl", sys.argv[1], arr_window_cl, 5, 5, 5)
	wrp.saveImage(sys.argv[2])
	
	# vglCl3dCopy
	wrp.vglCl3dCopy("../CL/vglCl3dCopy.cl", sys.argv[1])
	wrp.saveImage(sys.argv[2])

	# vglCl3dDilate
	wrp.vglCl3dDilate("../CL/vglCl3dDilate.cl", sys.argv[1], arr_window_cl, window_x, window_y, window_z)
	wrp.saveImage(sys.argv[2])

	# vglCl3dErode
	wrp.vglCl3dErode("../CL/vglCl3dErode.cl", sys.argv[1], arr_window_cl, window_x, window_y, window_z)
	wrp.saveImage(sys.argv[2])

	# vglCl3dMax
	wrp.vglCl3dMax("../CL/vglCl3dMax.cl", sys.argv[1], sys.argv[2])
	wrp.saveImage(sys.argv[3])

	# vglCl3dMin
	wrp.vglCl3dMin("../CL/vglCl3dMin.cl", sys.argv[1], sys.argv[2])
	wrp.saveImage(sys.argv[3])

	# vglCl3dNot
	wrp.vglCl3dNot("../CL/vglCl3dNot.cl", sys.argv[1])
	wrp.saveImage(sys.argv[2])

	# vglCl3dSub
	wrp.vglCl3dSub("../CL/vglCl3dSub.cl", sys.argv[1], sys.argv[2])
	wrp.saveImage(sys.argv[3])

	# vglCl3dSum
	wrp.vglCl3dSum("../CL/vglCl3dSum.cl", sys.argv[1], sys.argv[2])
	wrp.saveImage(sys.argv[3])

	# vglCl3dThreshold
	wrp.vglCl3dThreshold("../CL/vglCl3dThreshold.cl", sys.argv[1])
	wrp.saveImage(sys.argv[2])

	# vglClBlurSq3
	wrp.vglClBlurSq3("../CL/vglClBlurSq3.cl", sys.argv[1])
	wrp.saveImage(sys.argv[2])

	# vglClConvolution
	arr_window = np.ones((10,10), np.float32) * (1/100)
	arr_window_cl = cl.Buffer(wrp.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=arr_window)
	wrp.vglClConvolution("../CL/vglClConvolution.cl", sys.argv[1], arr_window_cl, 10, 10)
	wrp.saveImage(sys.argv[2])

	# vglClCopy
	wrp.vglClCopy("../CL/vglClCopy.cl", sys.argv[1])
	wrp.saveImage(sys.argv[2])

	# vglClDilate
	wrp.vglClDilate("../CL/vglClDilate.cl", sys.argv[1], arr_window, window_x, window_y)
	wrp.saveImage(sys.argv[2])

	# vglClErode
	wrp.vglClErode("../CL/vglClErode.cl", sys.argv[1], arr_window, window_x, window_y)
	wrp.saveImage(sys.argv[2])

	# vglClMax
	wrp.vglClMax("../CL/vglClMax.cl", sys.argv[1], sys.argv[2], arr_window, window_x, window_y)
	wrp.saveImage(sys.argv[3])

	# vglClMin
	wrp.vglClMin("../CL/vglClMin.cl", sys.argv[1], sys.argv[2], arr_window, window_x, window_y)
	wrp.saveImage(sys.argv[3])
	
	# vglClInvert
	wrp.vglClInvert("../CL/vglClInvert.cl", sys.argv[1])
	wrp.saveImage(sys.argv[2])

	# vglClSub
	wrp.vglClSub("../CL/vglClSub.cl", sys.argv[1], sys.argv[2])
	wrp.saveImage(sys.argv[3])

	# vglClSum
	wrp.vglClSum("../CL/vglClSum.cl", sys.argv[1], sys.argv[2])
	wrp.saveImage(sys.argv[3])

	"""

	# vglClThreshold
	wrp.vglClThreshold("../CL/vglClThreshold.cl", sys.argv[1])
	wrp.saveImage(sys.argv[2])

	"""
	# vglClNdConvolution
	wrp.vglClNdConvolution("../CL_ND/vglClNdConvolution.cl", sys.argv[1], vl.VGL_STREL_CROSS(), 2)
	wrp.saveImage(sys.argv[2])
	
	# vglClNdCopy
	wrp.vglClNdCopy("../CL_ND/vglClNdCopy.cl", sys.argv[1])
	wrp.saveImage(sys.argv[2])
	"""