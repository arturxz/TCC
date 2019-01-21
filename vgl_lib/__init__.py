# __init__.py

from .struct_sizes 		import struct_sizes
from .opencl_context	import opencl_context, VglClContext
from .vglShape 			import VglShape
from .vglStrEl 			import VglStrEl

# VGLIMAGE METHODS
from .vglImage import VglImage
from .vglImage import vglLoadImage
from .vglImage import vglImage3To4Channels
from .vglImage import vglImage4To3Channels
from .vglImage import vglSaveImage
from .vglImage import create_vglShape
from .vglImage import rgb_to_rgba
from .vglImage import rgba_to_rgb

# VGLCLIMAGE METHODS
from .vglClImage import vglClInit
from .vglClImage import vglClUpload
from .vglClImage import vglClDownload
from .vglClImage import cl_channel_type
from .vglClImage import cl_channel_order
from .vglClImage import get_ocl
from .vglClImage import set_ocl
from .vglClImage import get_ocl_context
from .vglClImage import get_similar_oclPtr_object
from .vglClImage import create_blank_image_as
from .vglClImage import get_vglstrel_opencl_buffer
from .vglClImage import get_vglshape_opencl_buffer

# OPENCL CONTEXT METHODS IMPORT
from .vglContext import vglIsContextValid
from .vglContext import vglIsContextUnique
from .vglContext import vglIsInContext
from .vglContext import vglAddContext
from .vglContext import vglSetContext
from .vglContext import vglCheckContext
from .vglContext import vglCheckContextForOutput

# CONSTANT RETURNS
from .vglConst import VGL_WIN_X0
from .vglConst import VGL_WIN_X1
from .vglConst import VGL_WIN_DX
from .vglConst import VGL_WIN_Y0
from .vglConst import VGL_WIN_Y1
from .vglConst import VGL_WIN_DY
from .vglConst import VGL_MIN_WINDOW_SPLIT
from .vglConst import VGL_DEFAULT_WINDOW_SPLIT
from .vglConst import VGL_MAX_WINDOW_SPLIT
from .vglConst import VGL_MAX_WINDOWS
from .vglConst import VGL_SHAPE_NCHANNELS 
from .vglConst import VGL_SHAPE_WIDTH 
from .vglConst import VGL_SHAPE_HEIGHT
from .vglConst import VGL_SHAPE_LENGTH
from .vglConst import VGL_MAX_DIM
from .vglConst import VGL_ARR_SHAPE_SIZE
from .vglConst import VGL_ARR_CLSTREL_SIZE
from .vglConst import VGL_STREL_CUBE
from .vglConst import VGL_STREL_CROSS
from .vglConst import VGL_STREL_GAUSS
from .vglConst import VGL_STREL_MEAN
from .vglConst import VGL_IMAGE_3D_IMAGE
from .vglConst import VGL_IMAGE_2D_IMAGE
from .vglConst import VGL_BLANK_CONTEXT
from .vglConst import VGL_RAM_CONTEXT
from .vglConst import VGL_GL_CONTEXT
from .vglConst import VGL_CUDA_CONTEXT
from .vglConst import VGL_CL_CONTEXT
from .vglConst import VGL_ERROR
from .vglConst import IMAGE_CL_OBJECT
from .vglConst import IMAGE_ND_ARRAY