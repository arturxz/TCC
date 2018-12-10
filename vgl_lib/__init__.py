# __init__.py

from .structSizes 	import StructSizes
from .vglImage 		import VglImage
from .vglClContext	import VglClContext
from .vglShape 		import VglShape
from .vglStrEl 		import VglStrEl

# CONSTANT RETURNS
from .vglConst import VGL_SHAPE_NCHANNELS, VGL_SHAPE_WIDTH, VGL_SHAPE_HEIGHT, VGL_SHAPE_LENGTH, VGL_MAX_DIM, VGL_ARR_SHAPE_SIZE, VGL_ARR_CLSTREL_SIZE, VGL_STREL_CUBE, VGL_STREL_CROSS, VGL_STREL_GAUSS, VGL_STREL_MEAN, VGL_IMAGE_3D_IMAGE, VGL_IMAGE_2D_IMAGE, VGL_BLANK_CONTEXT, VGL_RAM_CONTEXT, VGL_GL_CONTEXT, VGL_CUDA_CONTEXT, VGL_CL_CONTEXT