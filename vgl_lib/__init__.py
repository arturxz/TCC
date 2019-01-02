# __init__.py

from .structSizes 	import StructSizes
from .vglImage 		import VglImage
from .openclContext	import OpenclContext, VglClContext
from .vglShape 		import VglShape
from .vglStrEl 		import VglStrEl

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
from .vglConst import IMAGE_CL_OBJECT
from .vglConst import IMAGE_ND_ARRAY