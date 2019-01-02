"""
	AS PYTHON DOESN'T HAVE CONSTANT DECLARATION, THE NEXT METHODS
	RETURN THE VALUES WHO NEED CONSTANT BEHAVIOR.
"""
# VGL_WIN DEFINES [pack64 branch]
def VGL_WIN_X0():
	return -1

def VGL_WIN_X1():
	return 1

def VGL_WIN_DX():
	return VGL_WIN_X1() - VGL_WIN_X0()

def VGL_WIN_Y0():
	return -1

def VGL_WIN_Y1():
	return 1

def VGL_WIN_DY():
	return VGL_WIN_Y1() - VGL_WIN_Y0()

def VGL_MIN_WINDOW_SPLIT():
	return 1

def VGL_DEFAULT_WINDOW_SPLIT():
	return 2

def VGL_MAX_WINDOW_SPLIT():
	return 4

def VGL_MAX_WINDOWS():
	return VGL_MAX_WINDOW_SPLIT() * VGL_MAX_WINDOW_SPLIT()

# VGL CONSTANTS
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

# vglStrEl WINDOW MODES
def VGL_STREL_CUBE():
	return 1

def VGL_STREL_CROSS():
	return 2

def VGL_STREL_GAUSS():
	return 3

def VGL_STREL_MEAN():
	return 4

# IMAGE DIMENSIONS CONSTANTS
def VGL_IMAGE_3D_IMAGE():
	return 3

def VGL_IMAGE_2D_IMAGE():
	return 2

# CONTEXT CONSTANTS
def VGL_BLANK_CONTEXT():
	return 0

def VGL_RAM_CONTEXT():
	return 1

def VGL_GL_CONTEXT():
	return 2

def VGL_CUDA_CONTEXT():
	return 4

def VGL_CL_CONTEXT():
	return 8

# PYTHON-SPECIFIC IMAGE MANIPULATION MODE
def IMAGE_CL_OBJECT():
	return 0

def IMAGE_ND_ARRAY():
	return 1