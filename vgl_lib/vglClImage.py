from skimage import io
import pyopencl as cl
import numpy as np
import sys

# VISIONGL IMPORTS
import vgl_lib as vl

"""
	VARIABLE EQUIVALENT TO cl IN vglClImage.
"""
ocl = None

"""
	EQUIVALENT TO vglClInit METHOD, FOUND ON
	vglClImage.vglClInit().
"""
def vglClInit():
	ocl = vl.opencl_context().get_vglClContext_attributes()

"""
	EQUIVALENT TO vglClImage.vglClUpload()
	TREATING ACCORDING TO img_manipulation_mode.
"""
def vglClUpload(img):
	if( img.img_manipulation_mode == vl.IMAGE_CL_OBJECT() ):
		img.vglClImageUpload(img)
	elif( img.img_manipulation_mode == vl.IMAGE_ND_ARRAY() ):
		img.vglClNdImageUpload(img)
		
		vl.vglAddContext(img, vl.VGL_CL_CONTEXT())

"""
	EQUIVALENT TO vglClImage.vglClDownload()
	TREATING ACCORDING TO img_manipulation_mode.
"""
def vglClDownload(img):
	if( img.img_manipulation_mode is vl.IMAGE_CL_OBJECT() ):
		img.vglClImageDownload(img)
	elif( img.img_manipulation_mode is vl.IMAGE_ND_ARRAY() ):
		img.vglClNdImageDownload(img)

	vl.vglAddContext(img, vl.VGL_RAM_CONTEXT())

"""
	IT TAKES THE RAM-SIDE IMAGE, ALLOCATES THE
	DEVICE-SIDE MEMORY, CONSTRUCTS THE IMAGE OBJECT
	AND UPLOADS THE IMAGE OBJECT TO THE DEVICE.
"""
def vglClImageUpload(img):
	# IMAGE VARS
	print("Uploading image to device.")
	if( img.getVglShape().getNFrames() == 1 ):
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeight(), 1 )
		shape  = ( img.getVglShape().getWidth(), img.getVglShape().getHeight() )

		mf = cl.mem_flags
		imgFormat = cl.ImageFormat(img.get_toDevice_channel_order(), img.get_toDevice_dtype())
		img.img_device = cl.Image(ocl.ctx, mf.READ_ONLY, imgFormat, shape)
	elif( img.getVglShape().getNFrames() > 1 ):
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeight(), img.getVglShape().getNFrames() )
		shape = ( img.getVglShape().getWidth(), img.getVglShape().getHeight(), img.getVglShape().getNFrames() )

		mf = cl.mem_flags
		imgFormat = cl.ImageFormat(img.get_toDevice_channel_order(), img.get_toDevice_dtype())
		img.img_device = cl.Image(ocl.ctx, mf.READ_ONLY, imgFormat, shape)
	else:
		print("VglImage NFrames wrong. NFrames returns:", img.getVglShape().getNFrames() )
		return

	# COPYING NDARRAY IMAGE TO OPENCL IMAGE OBJECT
	cl.enqueue_copy(ocl.queue, img.img_device, img.img_ram, origin=origin, region=region, is_blocking=True)

"""
	THIS METHOD TAKES THE DEVICE-SIDE OPENCL IMAGE AND
	COPY IT BACK TO THE RAM-SIDE. IN THE END, THE METHOD
	CALL THE vglShape CONSTRUCTOR, TO ATUALIZE THE vglShape
	OBJECT.
"""
def vglClImageDownload(img):
	# MAKE IMAGE DOWNLOAD HERE
	print("Downloading Image from device.")

	if( img.getVglShape().getNFrames() == 1 ):
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeight(), 1 )
		totalSize = img.getVglShape().getHeight() * img.getVglShape().getWidth() * img.getVglShape().getNChannels()

		buffer = np.zeros(totalSize, img.img_ram.dtype)
		cl.enqueue_copy(ocl.queue, buffer, img.img_device, origin=origin, region=region, is_blocking=True)

		if( img.getVglShape().getNChannels() == 1 ):
			buffer = np.frombuffer( buffer, img.img_ram.dtype ).reshape( img.getVglShape().getHeight(), img.getVglShape().getWidth() )
		elif( (img.getVglShape().getNChannels() == 3) or (img.getVglShape().getNChannels() == 4) ):
			buffer = np.frombuffer( buffer, img.img_ram.dtype ).reshape( img.getVglShape().getHeight(), img.getVglShape().getWidth(), img.getVglShape().getNChannels() )
	elif( img.getVglShape().getNFrames() > 1 ):
		pitch = (0, 0)
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeight(), img.getVglShape().getNFrames() )
		totalSize = img.getVglShape().getHeight() * img.getVglShape().getWidth() * img.getVglShape().getNFrames()

		buffer = np.zeros(totalSize, img.img_ram.dtype)
		cl.enqueue_copy(ocl.queue, buffer, img.img_device, origin=origin, region=region, is_blocking=True)


		if( img.getVglShape().getNChannels() == 1 ):
			buffer = np.frombuffer( buffer, img.img_ram.dtype ).reshape( img.getVglShape().getNFrames(), img.getVglShape().getHeight(), img.getVglShape().getWidth() )
		elif( (img.getVglShape().getNChannels() == 3) or (img.getVglShape().getNChannels() == 4) ):
			buffer = np.frombuffer( buffer, img.img_ram.dtype ).reshape( img.getVglShape().getNFrames(), img.getVglShape().getHeight(), img.getVglShape().getWidth(), img.getVglShape().getNChannels() )

	img.img_ram = buffer
	img.create_vglShape()

"""
	IT TAKES THE RAM-SIDE IMAGE, ALLOCATES THE
	DEVICE-SIDE MEMORY, AND PASSES THE ND-ARRAY
	TO THE DEVICE.
"""
def vglClNdImageUpload(img):
	print("NdArray image Upload")
		
	# CREATING DEVICE POINTER AND COPYING HOST TO DEVICE
	mf = cl.mem_flags
	img.img_device = cl.Buffer(ocl.ctx, mf.READ_ONLY, img.get_ram_image().nbytes)
	cl.enqueue_copy(ocl.queue, img.img_device, img.get_ram_image().tobytes(), is_blocking=True)


"""
	THIS METHOD TAKES THE DEVICE-SIDE OPENCL BUFFER AND
	COPY IT BACK TO THE RAM-SIDE. THEN, THE METHOD CALLS
	THE vglShape CONSTRUCTOR, TO ATUALIZE THE vglShape
	OBJECT.
"""
def vglClNdImageDownload(img):
	print("NdArray image Download")

	cl.enqueue_copy(ocl.queue, img.img_ram, img.img_device)
	img.create_vglShape()
