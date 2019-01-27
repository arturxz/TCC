from skimage import io
import pyopencl as cl
import numpy as np
import sys

# VISIONGL IMPORTS
import vgl_lib as vl

# TO INFER TYPE TO THE VARIABLE
from typing import Union

"""
	ocl AND ocl_context ARE GLOBAL VARIABLES.
	ocl IS EQUIVALENT TO cl ON ORIGINAL vglClImage.
"""
ocl: Union[vl.VglClContext] = None
ocl_context: Union[vl.opencl_context] = None

"""
	struct_sizes IS A PYTHON-EXCLUSIVE GLOBAL VARIABLE.

	THIS VARIABLE CAN BE USED TO STORE THE WAY CURRENT
	OPENCL-DEVICE ORGANIZES THE vglClStrEl AND vglClShape
	DATA. WITH THE ORGANIZATION INFORMATION, THE BITWISE 
	PROCESS CAN BE DONE.
"""
struct_sizes: Union[np.ndarray] = None

"""
	EQUIVALENT TO vglClInit METHOD, FOUND ON
	vglClImage.vglClInit().
"""
def vglClInit():
	global ocl
	global ocl_context
	global struct_sizes

	ocl_context = vl.opencl_context()
	ocl = ocl_context.get_vglClContext_attributes()
	ss = vl.struct_sizes()
	struct_sizes = ss.get_struct_sizes()

	ss = None

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
	mf = cl.mem_flags

	# IMAGE VARS
	print("Uploading image to device.")
	if( img.getVglShape().getNFrames() == 1 ):
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeight(), 1 )
		shape  = ( img.getVglShape().getWidth(), img.getVglShape().getHeight() )

		imgFormat = cl.ImageFormat(vl.cl_channel_order(img), vl.cl_channel_type(img))
		img.oclPtr = cl.Image(ocl.context, mf.READ_ONLY, imgFormat, shape)
	elif( img.getVglShape().getNFrames() > 1 ):
		origin = ( 0, 0, 0 )
		region = ( img.getVglShape().getWidth(), img.getVglShape().getHeight(), img.getVglShape().getNFrames() )
		shape = ( img.getVglShape().getWidth(), img.getVglShape().getHeight(), img.getVglShape().getNFrames() )

		imgFormat = cl.ImageFormat(vl.cl_channel_order(img), vl.cl_channel_type(img))
		img.oclPtr = cl.Image(ocl.context, mf.READ_ONLY, imgFormat, shape)
	else:
		print("vglClImageUpload: VglImage NFrames wrong. NFrames returns:", img.getVglShape().getNFrames() )
		exit()

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
	mf = cl.mem_flags

	print("vglClNdImageUpload: Uploading to cl.Buffer")	
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
	print("vglClNdImageDownload: Downloading from cl.Buffer")

	cl.enqueue_copy(ocl.commandQueue, img.get_ipl(), img.get_oclPtr())
	vl.create_vglShape(img)

"""
	METHOD CURRENTLY NOT IN USE

	EQUIVALENT TO vglClImage.vglClCheckError
	THIS METHOD VERIFIES IF IS REALLY AN ERROR.
	IF IS, IT SHOWS THE ERROR ID AND PRINTS THE CAUSE. 
"""
def vglClCheckError(error, name):
	if(error < vl.CL_SUCCESS() and error >= vl.CL_MIN_ERROR()):
		print("Error", error, vl.vglClErrorMessages()[error], "while doing the following operation:")
		print(name)
		exit(error)

"""
	PYTHON-EXCLUSIVE METHODS
"""

"""
	RETURNS THE vl.VglClContext OBJECT
"""
def get_ocl():
	return ocl

"""
	RETURNS THE vl.opencl_context OBJECT
"""
def get_ocl_context():
	return ocl_context

"""
	SETS THE vl.VglClContext OBJECT
"""
def set_ocl(ctx):
	global ocl
	if( isinstance(ctx, vl.VglClContext) ):
		ocl = ctx
	else:
		print("Error! not VglClContext object!")

"""
	RETURNS THE IMAGE CHANNEL TYPE
"""
def cl_channel_type(img):
	oclPtr_dtype = None
	if( img.ipl.dtype == np.uint8 ):
		oclPtr_dtype = cl.channel_type.UNORM_INT8
		print("cl_channel_type: 8bit Channel Size!")
	elif( img.ipl.dtype == np.uint16 ):
		oclPtr_dtype = cl.channel_type.UNORM_INT16
		print("cl_channel_type: 16bit Channel Size!")

	return oclPtr_dtype

"""
	RETURNS THE IMAGE CHANNEL ORDER
	(OR QUANTITY OF CHANNELS)
"""
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

"""
	RETURNS A cl.Buffer or cl.Image OBJECT
	WITH THE SAME PROPERTIES OF THE img SENT
	AS ARGUMENT.

	IT IS USED TO CREATE A OUTPUT IMAGE WITH
	THE SAME SIZE AS THE INPUT IMAGE, FOR 
	EXAMPLE.
"""
def get_similar_oclPtr_object(img):
	global ocl
	mf = cl.mem_flags

	opencl_device: Union[cl.Image, cl.Buffer] = None

	if( isinstance(img.get_oclPtr(), cl.Image) ):
		print("get_similar_oclPtr_object: oclPtr is cl.Image.")
		imgFormat = cl.ImageFormat(vl.cl_channel_order(img), vl.cl_channel_type(img))
		opencl_device = cl.Image(ocl.context, mf.WRITE_ONLY, imgFormat, img.get_oclPtr().shape )
	elif isinstance(img.get_oclPtr(), cl.Buffer):
		print("get_similar_oclPtr_object: oclPtr is cl.Buffer.")
		opencl_device = cl.Buffer(ocl.context, mf.WRITE_ONLY, img.get_ipl().nbytes)
	
	return opencl_device

"""
	IT RETURNS A vl.Image OBJECT, WITH SIMILAR
	PROPERTIES AS THE img SENT AS ARGUMENT.

	img.ipl IS NOT SET, BUT OTHER PROPERTIES ARE.
	THE img.inContext IS SET AS VGL_BLANK_CONTEXT().
"""
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

"""
"""
def get_struct_sizes():
	global struct_sizes
	if( not isinstance(struct_sizes, np.ndarray) ):
		vl.vglClInit()
		print("get_struct_sizes: Warning! get_struct_sizes before vglClInit. Calling vglClInit now...")

	return struct_sizes

"""
	RETURNS A OPENCL BUFFER WITH DATA 
	OF VglClStrel TO KERNELS TO READ
"""
def get_vglstrel_opencl_buffer(strel):
	global ocl
	buf = cl.Buffer(ocl.context, cl.mem_flags.READ_ONLY, strel.nbytes)
	cl.enqueue_copy(ocl.commandQueue, buf, strel.tobytes(), is_blocking=True)
	return buf

"""
	RETURNS A OPENCL BUFFER WITH DATA 
	OF VglClShape TO KERNELS TO READ
"""
def get_vglshape_opencl_buffer(shape):
	global ocl
	buf = cl.Buffer(ocl.context, cl.mem_flags.READ_ONLY, shape.nbytes)
	cl.enqueue_copy(ocl.commandQueue, buf, shape.tobytes(), is_blocking=True)
	return buf

