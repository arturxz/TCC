"""
	AS PYTHON DOESN'T HAVE CONSTANT DECLARATION, THE NEXT METHODS
	RETURN THE VALUES WHO NEED CONSTANT BEHAVIOR.
"""

def VGL_SHAPE_NCHANNELS():
	return 0

def VGL_SHAPE_WIDTH():
	return 1

def VGL_SHAPE_HEIGHT():
	return 2

def VGL_SHAPE_LENGTH():
	return 3

def VGL_MAX_DIM():
	return 10

def VGL_ARR_SHAPE_SIZE():
	return VGL_MAX_DIM()+1

def VGL_ARR_CLSTREL_SIZE():
	return 256

def VGL_STREL_CUBE():
	return 1

def VGL_STREL_CROSS():
	return 2

def VGL_STREL_GAUSS():
	return 3

def VGL_STREL_MEAN():
	return 4

def VGL_IMAGE_3D_IMAGE():
	return 0

def VGL_IMAGE_2D_IMAGE():
	return 1