from skimage import io
import pyopencl as cl
import numpy as np
import sys

# VISIONGL IMPORTS
import vgl_lib as vl

"""
	img:
		is the input image
	cl_shape:
		3D Images:
			The OpenCL's default is to be (img_width, img_height, img_depht)
		2D Images:
			The The OpenCL's default is to be (img_width, img_height)
	cl_pitch:
		3D Images (needed):
			The OpenCL's default is to be (img_width*bytes_per_pixel, img_height*img_width*bytes_per_pixel)
			and it is assumed when pitch=(0, 0) is given
		2D Images (optional):
			The OpenCL's default is to be (img_width*bytes_per_pixel)
			and it is assumed when pitch=(0) is given
	cl_origin
		Is the origin of the image, where to start copying the image.
		2D images must have 0 in the z-axis
	cl_region
		Is where to end the copying of the image.
		2D images must have 1 in the z-axis
"""

"""
	PYTHON'S VGLIMAGE IS SLIGHTLY DIFFERENT  FROM ITS EQUIVALENT.
	ndim DEFAULTS TO 2D DIMENTION IMAGE
	
	img_mode IS HOW THE IMAGE WILL BE TREATED. IF WILL USE THE
			 ND-ARRAY IMAGE AND PASS THE IMAGE DATA TO THE DEVICE
			 AS A BUFFER OF DATA OR IF WILL CREATE A OPENCL IMAGE
			 OBJECT TO PASS ALL THE NEEDED DATA. IT DEFAULTS TO
			 THE SECOND ONE (OPENCL IMAGE OBJECT).
	
	ALL THE CONSTANTS CAN BE FOUND IN vglConst.py file.
"""
class VglImage(object):
	def __init__(self, imgPath="", ndim=None, img_mode=None ):
		# IF THE IMAGE TYPE IS NOT SPECIFIED, A 2D IMAGE WILL BE ASSUMED
		# INICIALIZING DATA
		self.inContext = 0
		self.filename = imgPath
		self.ndim = ndim
		self.depht = 0
		self.nChannels = 0
		self.has_mipmap = 0
		self.clForceAsBuf = img_mode

		# PYTHON-EXCLUSIVE DATA
		self.ipl = None
		self.oclPtr = None

		if( self.clForceAsBuf is None ):
			self.clForceAsBuf = vl.IMAGE_CL_OBJECT()
		elif( not((self.clForceAsBuf is vl.IMAGE_CL_OBJECT() )
			   or (self.clForceAsBuf is vl.IMAGE_ND_ARRAY() ) ) ):
			print("ERROR! UNEXISTENT IMAGE MODE! YOU'LL NEED TO INSTANTIATE AGAIN!")

		if(self.ndim is None):
			self.ndim = vl.VGL_IMAGE_2D_IMAGE()
			print("Creating 2D Image!")
		elif(self.ndim is vl.VGL_IMAGE_3D_IMAGE()):
			print("Creating 3D Image!")

		# OPENING IMAGE
		self.vglLoadImage()
	
	"""
		THIS METHOD INITIATES THE vglShape OBJECT
		TO THE IMAGE. IT LOOKS IF IS A 2D OR A 3D IMAGE
		AND CONSTRUCT THE vglShape OBJECT.

		IN EVERY CHANGE IN THE IMAGE, THIS METHOD MUST BE
		CALLED, TO ASSURES THAT THE vglShape IS SYNCED WITH
		THE ACTUAL IMAGE.
	"""
	def create_vglShape(self):
		if(self.ipl is not None):
			print("The image was founded. Creating vglShape.")

			self.vglshape = vl.VglShape()
			if( self.ndim == vl.VGL_IMAGE_2D_IMAGE() ):
				print("2D Image")
				if( len(self.ipl.shape) == 2 ):
					# SHADES OF GRAY IMAGE
					print("VglImage LUMINANCE")
					self.vglshape.constructor2DShape(1, self.ipl.shape[1], self.ipl.shape[0])
				elif(len(self.ipl.shape) == 3):
					# MORE THAN ONE COLOR CHANNEL
					print("VglImage RGB")
					self.vglshape.constructor2DShape(self.ipl.shape[2], self.ipl.shape[1], self.ipl.shape[0])
			elif( self.ndim == vl.VGL_IMAGE_3D_IMAGE() ):
				print("3D Image")
				if( len(self.ipl.shape) == 3 ):
					# SHADES OF GRAY IMAGE
					print("VglImage LUMINANCE")
					self.vglshape.constructor3DShape( 1, self.ipl.shape[2], self.ipl.shape[1], self.ipl.shape[0] )
				elif(len(self.ipl.shape) == 4):
					# MORE THAN ONE COLOR CHANNEL
					print("VglImage RGB")
					self.vglshape.constructor3DShape( self.ipl.shape[3], self.ipl.shape[2], self.ipl.shape[1], self.ipl.shape[0] )
		
			#self.img_sync = False
			#self.last_changed_host = True
			#self.last_changed_device = False
		else:
			print("Impossible to create a vglImage object. ram_image is None.")

	"""
		EQUIVALENT TO:
			vglImage.vglLoadImage
			vglImage.vglLoad3dImage
			vglImage.vglLoadNdImage
			vglImage.vglLoadPgm

		THIS METHOD READS THE IMAGE PATH AND OPEN
		IT AS A NUMPY NDARRAY. THROWS A ERROR MESSAGE
		IF THE PATH IS INCORRECT. BUILD THE MAKE THE CALL
		TO CONSTRUCT THE vglShape OBJECT. 
	"""
	def vglLoadImage(self):
		try:
			if( self.filename is "" ):
				self.inContext = 0
				self.ipl = np.zeros((1, 1), np.uint8)
				print("Making a BLANK_IMAGE")
			else:
				self.ipl = io.imread(self.filename)
				vl.vglAddContext(self, vl.VGL_RAM_CONTEXT())
		except FileNotFoundError as fnf:
			print("vglCreateImage: Error loading image from file:", self.filename)    
			print(str(fnf))
		except Exception as e:
			print("Unrecognized error:")
			print(str(e))

		self.create_vglShape()
	
	"""
		EQUIVALENT TO vglImage.vglImage3To4Channels()
	"""
	def vglImage3To4Channels(self):
		self.rgb_to_rgba()

	"""
		EQUIVALENT TO vglImage.vglImage4To3Channels()
	"""
	def vglImage4To3Channels(self):
		self.rgba_to_rgb()
	
	def getVglShape(self):
		return self.vglshape
		
	"""
		EQUIVALENT TO DIFFERENT IMAGE SAVE
		METHODS IN vglImage.cpp
	"""
	def img_save(self, name):
		print("Saving Picture in Hard Drive")
		io.imsave(name, self.ipl)

	"""
		EQUIVALENT TO vglImage.3To4Channels()
	"""
	def rgb_to_rgba(self):
		print("[RGB -> RGBA]")
		ipl_rgba = np.empty((self.vglshape.getHeight(), self.vglshape.getWidth(), 4), self.ipl.dtype)

		ipl_rgba[:,:,0] = self.ipl[:,:,0]
		ipl_rgba[:,:,1] = self.ipl[:,:,1]
		ipl_rgba[:,:,2] = self.ipl[:,:,2]
		ipl_rgba[:,:,3] = 255

		self.ipl = ipl_rgba
		self.create_vglShape()

	"""
		EQUIVALENT TO vglImage.3To4Channels()
	"""
	def rgba_to_rgb(self):
		print("[RGBA -> RGB]")
		if( (self.ipl[0,0,:].size < 4) | (self.ipl[0,0,:].size > 4) ):
			print("IMAGE IS NOT RGBA")
		else:
			ipl_rgb = np.empty((self.vglshape.getHeight(), self.vglshape.getWidth(), 3), self.ipl.dtype)
			ipl_rgb[:,:,0] = self.ipl[:,:,0]
			ipl_rgb[:,:,1] = self.ipl[:,:,1]
			ipl_rgb[:,:,2] = self.ipl[:,:,2]

			self.ipl = ipl_rgb
			self.create_vglShape()

	def get_similar_oclPtr_object(self, ctx, queue):

		if(self.ndim == vl.VGL_IMAGE_2D_IMAGE()):
			shape  = ( self.vglshape.getWidth(), self.vglshape.getHeight() )
			mf = cl.mem_flags
			imgFormat = cl.ImageFormat(vl.cl_channel_order(self), vl.cl_channel_type(self))
			img_copy = cl.Image(ctx, mf.WRITE_ONLY, imgFormat, shape)
		elif(self.ndim == vl.VGL_IMAGE_3D_IMAGE()):
			shape  = ( self.vglshape.getWidth(), self.vglshape.getHeight(), self.vglshape.getNFrames() )
			mf = cl.mem_flags
			imgFormat = cl.ImageFormat(vl.cl_channel_order(self), vl.cl_channel_type(self))
			img_copy = cl.Image(ctx, mf.WRITE_ONLY, imgFormat, shape)

		print("--> Orig:", self.get_oclPtr().width, self.get_oclPtr().height, self.get_oclPtr().depth)
		print("--> Copy:", img_copy.width, img_copy.height, img_copy.depth)

		return img_copy
	
	def set_oclPtr(self, img):
		if( isinstance(img, cl.Image) or isinstance(img, cl.Buffer) ):
			self.oclPtr = img
		else:
			print("Invalid object. cl.Image or cl.Buffer objects only.")
	
	def get_oclPtr(self):
		return self.oclPtr
	
	def get_ipl(self):
		return self.ipl
