from skimage import io
import pyopencl as cl
import numpy as np
import sys

# VISIONGL IMPORTS
import vgl_lib as vl

# TO INFER TYPE TO THE VARIABLE
from typing import Union

"""
	VARIABLE EQUIVALENT TO cl IN vglClImage.
"""
ocl: Union[vl.VglClContext] = None
ocl_context: Union[vl.opencl_context] = None

def get_ocl():
	return ocl

def get_ocl_context():
	return ocl_context

def set_ocl(ctx):
	global ocl
	if( isinstance(ctx, vl.VglClContext) ):
		ocl = ctx
	else:
		print("Error! not VglClContext object!")

"""
	EQUIVALENT TO vglClInit METHOD, FOUND ON
	vglClImage.vglClInit().
"""
def vglClInit():
	global ocl
	global ocl_context
	ocl_context = vl.opencl_context()
	ocl = ocl_context.get_vglClContext_attributes()

"""
	EQUIVALENT TO vglClImage.vglClUpload()
	TREATING ACCORDING TO clForceAsBuf.
"""
def vglClUpload(img):

	if( vl.vglIsInContext(img, vl.VGL_RAM_CONTEXT()) or vl.vglIsInContext(img, vl.VGL_BLANK_CONTEXT()) ):
		if( img.clForceAsBuf == vl.IMAGE_CL_OBJECT() ):
			vglClImageUpload(img)
		elif( img.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			vglClNdImageUpload(img)
		
		vl.vglAddContext(img, vl.VGL_CL_CONTEXT())
	else:
		print("vglClUpload: Error: image context is not in VGL_RAM_CONTEXT or VGL_BLANK_CONTEXT.")
		exit()

"""
	EQUIVALENT TO vglClImage.vglClDownload()
	TREATING ACCORDING TO clForceAsBuf.
"""
def vglClDownload(img):
	if( vl.vglIsInContext(img, vl.VGL_CL_CONTEXT()) ):
		if( img.clForceAsBuf == vl.IMAGE_CL_OBJECT() ):
			vglClImageDownload(img)
		elif( img.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
			vglClNdImageDownload(img)

		vl.vglAddContext(img, vl.VGL_RAM_CONTEXT())
	else:
		print("vglClDownload: Error: image context is not in VGL_CL_CONTEXT.")
		exit()

"""
	IT TAKES THE RAM-SIDE IMAGE, ALLOCATES THE
	DEVICE-SIDE MEMORY, CONSTRUCTS THE IMAGE OBJECT
	AND UPLOADS THE IMAGE OBJECT TO THE DEVICE.
"""
def vglClImageUpload(img):
	global ocl

	# IMAGE VARS
	print("Uploading image to device.")
	if( img.getVglShape().getNFrames() == 1 ):
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeight(), 1 )
		shape  = ( img.getVglShape().getWidth(), img.getVglShape().getHeight() )

		mf = cl.mem_flags
		imgFormat = cl.ImageFormat(cl_channel_order(img), cl_channel_type(img))
		img.oclPtr = cl.Image(ocl.context, mf.READ_ONLY, imgFormat, shape)
	elif( img.getVglShape().getNFrames() > 1 ):
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeight(), img.getVglShape().getNFrames() )
		shape = ( img.getVglShape().getWidth(), img.getVglShape().getHeight(), img.getVglShape().getNFrames() )

		mf = cl.mem_flags
		imgFormat = cl.ImageFormat(cl_channel_order(img), cl_channel_type(img))
		img.oclPtr = cl.Image(ocl.context, mf.READ_ONLY, imgFormat, shape)
	else:
		print("VglImage NFrames wrong. NFrames returns:", img.getVglShape().getNFrames() )
		return

	# COPYING NDARRAY IMAGE TO OPENCL IMAGE OBJECT
	cl.enqueue_copy(ocl.commandQueue, img.get_oclPtr(), img.get_ipl(), origin=origin, region=region, is_blocking=True)

"""
	THIS METHOD TAKES THE DEVICE-SIDE OPENCL IMAGE AND
	COPY IT BACK TO THE RAM-SIDE. IN THE END, THE METHOD
	CALL THE vglShape CONSTRUCTOR, TO ATUALIZE THE vglShape
	OBJECT.
"""
def vglClImageDownload(img):
	global ocl 

	# MAKE IMAGE DOWNLOAD HERE
	print("Downloading Image from device.")

	if( img.getVglShape().getNFrames() == 1 ):
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeight(), 1 )
		totalSize = img.getVglShape().getHeight() * img.getVglShape().getWidth() * img.getVglShape().getNChannels()

		buffer = np.zeros(totalSize, img.get_ipl().dtype)
		cl.enqueue_copy(ocl.commandQueue, buffer, img.get_oclPtr(), origin=origin, region=region, is_blocking=True)

		if( img.getVglShape().getNChannels() == 1 ):
			buffer = np.frombuffer( buffer, img.get_ipl().dtype ).reshape( img.getVglShape().getHeight(), img.getVglShape().getWidth() )
		elif( (img.getVglShape().getNChannels() == 3) or (img.getVglShape().getNChannels() == 4) ):
			buffer = np.frombuffer( buffer, img.get_ipl().dtype ).reshape( img.getVglShape().getHeight(), img.getVglShape().getWidth(), img.getVglShape().getNChannels() )
	elif( img.getVglShape().getNFrames() > 1 ):
		#pitch = (0, 0)
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeight(), img.getVglShape().getNFrames() )
		totalSize = img.getVglShape().getHeight() * img.getVglShape().getWidth() * img.getVglShape().getNFrames()

		buffer = np.zeros(totalSize, img.get_ipl().dtype)
		cl.enqueue_copy(ocl.commandQueue, buffer, img.get_oclPtr(), origin=origin, region=region, is_blocking=True)


		if( img.getVglShape().getNChannels() == 1 ):
			buffer = np.frombuffer( buffer, img.get_ipl().dtype ).reshape( img.getVglShape().getNFrames(), img.getVglShape().getHeight(), img.getVglShape().getWidth() )
		elif( (img.getVglShape().getNChannels() == 3) or (img.getVglShape().getNChannels() == 4) ):
			buffer = np.frombuffer( buffer, img.get_ipl().dtype ).reshape( img.getVglShape().getNFrames(), img.getVglShape().getHeight(), img.getVglShape().getWidth(), img.getVglShape().getNChannels() )

	img.ipl = buffer
	vl.create_vglShape(img)

"""
	IT TAKES THE RAM-SIDE IMAGE, ALLOCATES THE
	DEVICE-SIDE MEMORY, AND PASSES THE ND-ARRAY
	TO THE DEVICE.
"""
def vglClNdImageUpload(img):
	global ocl 

	print("NdArray image Upload")
		
	# CREATING DEVICE POINTER AND COPYING HOST TO DEVICE
	img.oclPtr = cl.Buffer(ocl.context, mf.READ_ONLY, img.get_ipl().nbytes)
	cl.enqueue_copy(ocl.commandQueue, img.get_oclPtr(), img.get_ipl().tobytes(), is_blocking=True)


"""
	THIS METHOD TAKES THE DEVICE-SIDE OPENCL BUFFER AND
	COPY IT BACK TO THE RAM-SIDE. THEN, THE METHOD CALLS
	THE vglShape CONSTRUCTOR, TO ATUALIZE THE vglShape
	OBJECT.
"""
def vglClNdImageDownload(img):
	global ocl
	print("NdArray image Download")

	cl.enqueue_copy(ocl.commandQueue, img.get_ipl(), img.get_oclPtr())
	img.create_vglShape()

def cl_channel_type(img):
	oclPtr_dtype = None
	if( img.ipl.dtype == np.uint8 ):
		oclPtr_dtype = cl.channel_type.UNORM_INT8
		print("8bit Channel Size!")
	elif( img.ipl.dtype == np.uint16 ):
		oclPtr_dtype = cl.channel_type.UNORM_INT16
		print("16bit Channel Size!")

	return oclPtr_dtype
	
def cl_channel_order(img):
	oclPtr_channel_order = None
	if( img.getVglShape().getNChannels() == 1 ):
		oclPtr_channel_order = cl.channel_order.LUMINANCE
	elif( img.getVglShape().getNChannels() == 2 ):
		oclPtr_channel_order = cl.channel_order.RG
	elif( img.getVglShape().getNChannels() == 3 ):
		oclPtr_channel_order = cl.channel_order.RGB
	elif( img.getVglShape().getNChannels() == 4 ):
		oclPtr_channel_order = cl.channel_order.RGBA
	
	return oclPtr_channel_order

def get_similar_oclPtr_object(img):
	global ocl
	mf = cl.mem_flags
	imgFormat = cl.ImageFormat(vl.cl_channel_order(img), vl.cl_channel_type(img))
	return cl.Image(ocl.context, mf.WRITE_ONLY, imgFormat, img.get_oclPtr().shape )

def create_blank_image_as(img):
	image = vl.VglImage(img.filename, img.ndim, img.clForceAsBuf)
	image.ipl		= np.asarray(img.ipl, img.ipl.dtype)
	image.shape		= img.shape
	image.vglShape	= img.vglShape
	image.depht		= img.depht
	image.nChannels	= img.nChannels
	image.has_mipmap= img.has_mipmap
	image.inContext	= vl.VGL_BLANK_CONTEXT()

	return image