from skimage import io
import pyopencl as cl
import numpy as np
import sys

# VISIONGL IMPORTS
import vgl_lib as vl

# TO INFER TYPE TO THE VARIABLE
from typing import Union

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
	VglImage is equivalent to vglImage object.
	isBuffer is a inicialization option that sets the
	clForceAsBuf. If it is not set in the image creation,
	is assumed that is a OpenCL Image object.

	ndim sets the number of dimensions of the image. If it
	is not set un the image creation, is assumed that is a
	2D image.

	ALL THE CONSTANTS CAN BE FOUND IN vglConst.py file.

	What python-version VglImage don't have right now?
		GLuint fbo equivalent; (GL not implemented)
		GLuint tex equivalent; (GL not implemented)
		void* cudaPtr equivalent; (CUDA not implemented)
		GLuint cudaPbo equivalent; (CUDA not implemented)
"""
class VglImage(object):
	def __init__(self, imgPath="", ndim=None, isBuffer=None ):
		# IF THE IMAGE TYPE IS NOT SPECIFIED, A 2D IMAGE WILL BE ASSUMED
		# INICIALIZING DATA
		self.ipl = None
		self.ndim = ndim
		self.shape = np.zeros((2*vl.VGL_MAX_DIM()), np.uint8)
		self.vglShape: Union[vl.vglShape] = None
		self.depht = 0
		self.nChannels = 0
		self.has_mipmap = 0
		self.oclPtr: Union[cl.Image, cl.Buffer] = None
		self.clForceAsBuf = isBuffer
		self.inContext = 0
		self.filename = imgPath

		if( self.clForceAsBuf is None ):
			self.clForceAsBuf = vl.IMAGE_CL_OBJECT()
		elif( not((self.clForceAsBuf is vl.IMAGE_CL_OBJECT() )
			   or (self.clForceAsBuf is vl.IMAGE_ND_ARRAY() ) ) ):
			print("ERROR! UNEXISTENT IMAGE MODE! YOU'LL NEED TO INSTANTIATE AGAIN!")

		if(self.ndim is None):
			self.ndim = vl.VGL_IMAGE_2D_IMAGE()
			print("Assuming 2D Image!")
		elif(self.ndim is vl.VGL_IMAGE_2D_IMAGE()):
			print("Creating 2D Image!")
		elif(self.ndim is vl.VGL_IMAGE_3D_IMAGE()):
			print("Creating 3D Image!")
	
	def getVglShape(self):
		return self.vglShape
	
"""
	THIS METHOD INITIATES THE vglShape OBJECT
	TO THE IMAGE. IT LOOKS IF IS A 2D OR A 3D IMAGE
	AND CONSTRUCT THE vglShape OBJECT.

	IN EVERY CHANGE IN THE IMAGE, THIS METHOD MUST BE
	CALLED, TO ASSURES THAT THE vglShape IS SYNCED WITH
	THE ACTUAL IMAGE.
"""
def create_vglShape(img):
	if(img.ipl is not None):
		print("The image is valid. Creating vglShape.")

		img.vglShape = vl.VglShape()
		if( img.ndim == vl.VGL_IMAGE_2D_IMAGE() ):
			print("2D Image")
			if( len(img.ipl.shape) == 2 ):
				# SHADES OF GRAY IMAGE
				print("VglImage LUMINANCE")
				img.vglShape.constructor2DShape(1, img.ipl.shape[1], img.ipl.shape[0])
			elif(len(img.ipl.shape) == 3):
				# MORE THAN ONE COLOR CHANNEL
				print("VglImage RGB")
				img.vglShape.constructor2DShape(img.ipl.shape[2], img.ipl.shape[1], img.ipl.shape[0])
		elif( img.ndim == vl.VGL_IMAGE_3D_IMAGE() ):
			print("3D Image")
			if( len(img.ipl.shape) == 3 ):
				# SHADES OF GRAY IMAGE
				print("VglImage LUMINANCE")
				img.vglShape.constructor3DShape( 1, img.ipl.shape[2], img.ipl.shape[1], img.ipl.shape[0] )
			elif(len(img.ipl.shape) == 4):
				# MORE THAN ONE COLOR CHANNEL
				print("VglImage RGB")
				img.vglShape.constructor3DShape( img.ipl.shape[3], img.ipl.shape[2], img.ipl.shape[1], img.ipl.shape[0] )
		else:
			print("Impossible to create a vglImage object. ram_image is None.")
	else:
		print("The image wasn't loaded. Please, load the image.")

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
def vglLoadImage(img, filename=""):
	try:
		if( img.filename is "" ):
			img.filename = filename
			img.ipl = np.zeros((1, 1), np.uint8)
			vl.vglSetContext(img, vl.VGL_BLANK_CONTEXT())
			print("VGL_BLANK_CONTEXT whith shape(1, 1) made")
		else:
			img.ipl = io.imread(img.filename)
			vl.vglAddContext(img, vl.VGL_RAM_CONTEXT())
			print("Image loaded! VGL_RAM_CONTEXT.")
	except FileNotFoundError as fnf:
		print("vglCreateImage: Error loading image from file:", img.filename)    
		print(str(fnf))
		exit()
	except Exception as e:
		print("Unrecognized error:")
		print(str(e))
		exit()

	create_vglShape(img)
	
"""
	EQUIVALENT TO vglImage.vglImage3To4Channels()
"""
def vglImage3To4Channels(img):
	rgb_to_rgba(img)

"""
	EQUIVALENT TO vglImage.vglImage4To3Channels()
"""
def vglImage4To3Channels(img):
	rgba_to_rgb(img)

"""
	EQUIVALENT TO DIFFERENT IMAGE SAVE
	METHODS IN vglImage.cpp
"""
def img_save(name, img):
	print("Saving Picture in Hard Drive")
	io.imsave(name, img.ipl)

"""
	EQUIVALENT TO vglImage.3To4Channels()
"""
def rgb_to_rgba(img):
	print("[RGB -> RGBA]")
	ipl_rgba = np.empty((self.vglshape.getHeight(), self.vglshape.getWidth(), 4), self.ipl.dtype)

	ipl_rgba[:,:,0] = img.ipl[:,:,0]
	ipl_rgba[:,:,1] = img.ipl[:,:,1]
	ipl_rgba[:,:,2] = img.ipl[:,:,2]
	ipl_rgba[:,:,3] = 255

	img.ipl = ipl_rgba
	create_vglShape(img)

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
