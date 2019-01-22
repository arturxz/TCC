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
	return VGL_SHAPE_D0()

def VGL_SHAPE_WIDTH():
	return VGL_SHAPE_D1()

def VGL_SHAPE_HEIGHT():
	return VGL_SHAPE_D2()

def VGL_SHAPE_LENGTH():
	return VGL_SHAPE_D3()

def VGL_4D():
	return 3

def VGL_MAX_DIM():
	return 10

def VGL_ARR_SHAPE_SIZE():
	return VGL_MAX_DIM()+1

def VGL_SHAPE_D0():
	return 0

def VGL_SHAPE_D1():
	return 1

def VGL_SHAPE_D2():
	return 2

def VGL_SHAPE_D3():
	return 3

def VGL_SHAPE_D4():
	return 4

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
"""
	VGL_ERROR WAS CREATED IN ORDER TO DIFFERENTIATE
	WHEN vglCheckContext(img, context) RETURNED 0 (ERROR) 
	OR VGL_BLANK_CONTEXT THAT WAS ALSO 0. THEN, IN THE
	METHOD vglCheckContext(), WHEN context IS NOT UNIQUE
	OR A ERROR OCCURS, IS RETURNED VGL_ERROR.
"""
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

def VGL_ERROR():
	return -1

# PYTHON-SPECIFIC IMAGE MANIPULATION MODE
def IMAGE_CL_OBJECT():
	return 0

def IMAGE_ND_ARRAY():
	return 1

import pyopencl as cl
import numpy as np

def VGL_PACK_SIZE_BITS_8():
	return 8

def VGL_PACK_SIZE_BYTES_8():
	return 1

def VGL_PACK_MAX_UINT_8():
	return 0xff

def VGL_PACK_CL_CONST_TYPE_8():
	return cl.channel_type.UNSIGNED_INT8

def VGL_PACK_CL_SHADER_TYPE_8():
	return np.uint8

def VGL_PACK_OUTPUT_SWAP_MASK_8():
	outputSwapMask = np.ndarray(32, VGL_PACK_CL_SHADER_TYPE_8)
	outputSwapMask[0] = VGL_PACK_CL_SHADER_TYPE_8()(0x80)
	outputSwapMask[1] = VGL_PACK_CL_SHADER_TYPE_8()(0x40)
	outputSwapMask[2] = VGL_PACK_CL_SHADER_TYPE_8()(0x20)
	outputSwapMask[3] = VGL_PACK_CL_SHADER_TYPE_8()(0x10)
	outputSwapMask[4] = VGL_PACK_CL_SHADER_TYPE_8()(0x08)
	outputSwapMask[5] = VGL_PACK_CL_SHADER_TYPE_8()(0x04)
	outputSwapMask[6] = VGL_PACK_CL_SHADER_TYPE_8()(0x02)
	outputSwapMask[7] = VGL_PACK_CL_SHADER_TYPE_8()(0x01)
	return outputSwapMask

def VGL_PACK_SIZE_BITS_32():
	return 32

def VGL_PACK_SIZE_BYTES_32():
	return 4

def VGL_PACK_MAX_UINT_32():
	return 0xffffffff

def VGL_PACK_CL_CONST_TYPE_32():
	return cl.channel_type.UNSIGNED_INT32

def VGL_PACK_CL_SHADER_TYPE_32():
	return np.uint32

def VGL_PACK_OUTPUT_SWAP_MASK_32():
	outputSwapMask = np.ndarray(32, VGL_PACK_CL_SHADER_TYPE_32)
	outputSwapMask[0] = VGL_PACK_CL_SHADER_TYPE_32()(0x80)
	outputSwapMask[1] = VGL_PACK_CL_SHADER_TYPE_32()(0x40)
	outputSwapMask[2] = VGL_PACK_CL_SHADER_TYPE_32()(0x20)
	outputSwapMask[3] = VGL_PACK_CL_SHADER_TYPE_32()(0x10)
	outputSwapMask[4] = VGL_PACK_CL_SHADER_TYPE_32()(0x08)
	outputSwapMask[5] = VGL_PACK_CL_SHADER_TYPE_32()(0x04)
	outputSwapMask[6] = VGL_PACK_CL_SHADER_TYPE_32()(0x02)
	outputSwapMask[7] = VGL_PACK_CL_SHADER_TYPE_32()(0x01)
	outputSwapMask[8] = VGL_PACK_CL_SHADER_TYPE_32()(0x8000)
	outputSwapMask[9] = VGL_PACK_CL_SHADER_TYPE_32()(0x4000)
	outputSwapMask[10] = VGL_PACK_CL_SHADER_TYPE_32()(0x2000)
	outputSwapMask[11] = VGL_PACK_CL_SHADER_TYPE_32()(0x1000)
	outputSwapMask[12] = VGL_PACK_CL_SHADER_TYPE_32()(0x0800)
	outputSwapMask[13] = VGL_PACK_CL_SHADER_TYPE_32()(0x0400)
	outputSwapMask[14] = VGL_PACK_CL_SHADER_TYPE_32()(0x0200)
	outputSwapMask[15] = VGL_PACK_CL_SHADER_TYPE_32()(0x0100)
	outputSwapMask[16] = VGL_PACK_CL_SHADER_TYPE_32()(0x800000)
	outputSwapMask[17] = VGL_PACK_CL_SHADER_TYPE_32()(0x400000)
	outputSwapMask[18] = VGL_PACK_CL_SHADER_TYPE_32()(0x200000)
	outputSwapMask[19] = VGL_PACK_CL_SHADER_TYPE_32()(0x100000)
	outputSwapMask[20] = VGL_PACK_CL_SHADER_TYPE_32()(0x08000000)
	outputSwapMask[21] = VGL_PACK_CL_SHADER_TYPE_32()(0x04000000)
	outputSwapMask[22] = VGL_PACK_CL_SHADER_TYPE_32()(0x02000000)
	outputSwapMask[23] = VGL_PACK_CL_SHADER_TYPE_32()(0x01000000)
	return outputSwapMask

def VGL_PACK_SIZE_BITS_64():
	return 64

def VGL_PACK_SIZE_BYTES_64():
	return 8

def VGL_PACK_MAX_UINT_64():
	return 0xffffffffffffffff

def VGL_PACK_CL_CONST_TYPE_64():
	return cl.channel_type.UNSIGNED_INT32

def VGL_PACK_CL_SHADER_TYPE_64():
	return np.uint64

def VGL_PACK_OUTPUT_SWAP_MASK_64():
	outputSwapMask = np.ndarray(64, VGL_PACK_CL_SHADER_TYPE_64)
	outputSwapMask[0] = VGL_PACK_CL_SHADER_TYPE_64()(0x80)
	outputSwapMask[1] = VGL_PACK_CL_SHADER_TYPE_64()(0x40)
	outputSwapMask[2] = VGL_PACK_CL_SHADER_TYPE_64()(0x20)
	outputSwapMask[3] = VGL_PACK_CL_SHADER_TYPE_64()(0x10)
	outputSwapMask[4] = VGL_PACK_CL_SHADER_TYPE_64()(0x08)
	outputSwapMask[5] = VGL_PACK_CL_SHADER_TYPE_64()(0x04)
	outputSwapMask[6] = VGL_PACK_CL_SHADER_TYPE_64()(0x02)
	outputSwapMask[7] = VGL_PACK_CL_SHADER_TYPE_64()(0x01)
	outputSwapMask[8] = VGL_PACK_CL_SHADER_TYPE_64()(0x8000)
	outputSwapMask[9] = VGL_PACK_CL_SHADER_TYPE_64()(0x4000)
	outputSwapMask[10] = VGL_PACK_CL_SHADER_TYPE_64()(0x2000)
	outputSwapMask[11] = VGL_PACK_CL_SHADER_TYPE_64()(0x1000)
	outputSwapMask[12] = VGL_PACK_CL_SHADER_TYPE_64()(0x0800)
	outputSwapMask[13] = VGL_PACK_CL_SHADER_TYPE_64()(0x0400)
	outputSwapMask[14] = VGL_PACK_CL_SHADER_TYPE_64()(0x0200)
	outputSwapMask[15] = VGL_PACK_CL_SHADER_TYPE_64()(0x0100)
	outputSwapMask[16] = VGL_PACK_CL_SHADER_TYPE_64()(0x800000)
	outputSwapMask[17] = VGL_PACK_CL_SHADER_TYPE_64()(0x400000)
	outputSwapMask[18] = VGL_PACK_CL_SHADER_TYPE_64()(0x200000)
	outputSwapMask[19] = VGL_PACK_CL_SHADER_TYPE_64()(0x100000)
	outputSwapMask[20] = VGL_PACK_CL_SHADER_TYPE_64()(0x08000000)
	outputSwapMask[21] = VGL_PACK_CL_SHADER_TYPE_64()(0x04000000)
	outputSwapMask[22] = VGL_PACK_CL_SHADER_TYPE_64()(0x02000000)
	outputSwapMask[23] = VGL_PACK_CL_SHADER_TYPE_64()(0x01000000)
	outputSwapMask[24] = VGL_PACK_CL_SHADER_TYPE_64()(0x0800000000)
	outputSwapMask[25] = VGL_PACK_CL_SHADER_TYPE_64()(0x0400000000)
	outputSwapMask[26] = VGL_PACK_CL_SHADER_TYPE_64()(0x0200000000)
	outputSwapMask[27] = VGL_PACK_CL_SHADER_TYPE_64()(0x0100000000)
	outputSwapMask[28] = VGL_PACK_CL_SHADER_TYPE_64()(0x080000000000)
	outputSwapMask[29] = VGL_PACK_CL_SHADER_TYPE_64()(0x040000000000)
	outputSwapMask[30] = VGL_PACK_CL_SHADER_TYPE_64()(0x020000000000)
	outputSwapMask[31] = VGL_PACK_CL_SHADER_TYPE_64()(0x010000000000)
	outputSwapMask[32] = VGL_PACK_CL_SHADER_TYPE_64()(0x08000000000000)
	outputSwapMask[33] = VGL_PACK_CL_SHADER_TYPE_64()(0x04000000000000)
	outputSwapMask[34] = VGL_PACK_CL_SHADER_TYPE_64()(0x02000000000000)
	outputSwapMask[35] = VGL_PACK_CL_SHADER_TYPE_64()(0x01000000000000)
	return outputSwapMask

def VGL_PACK_OUTPUT_DIRECT_MASK():
	outputDirectMask = np.ndarray(64, VGL_PACK_CL_SHADER_TYPE_64)
	outputDirectMask[0] = VGL_PACK_CL_SHADER_TYPE_64()(0x80)
	outputDirectMask[1] = VGL_PACK_CL_SHADER_TYPE_64()(0x40)
	outputDirectMask[2] = VGL_PACK_CL_SHADER_TYPE_64()(0x20)
	outputDirectMask[3] = VGL_PACK_CL_SHADER_TYPE_64()(0x10)
	outputDirectMask[4] = VGL_PACK_CL_SHADER_TYPE_64()(0x08)
	outputDirectMask[5] = VGL_PACK_CL_SHADER_TYPE_64()(0x04)
	outputDirectMask[6] = VGL_PACK_CL_SHADER_TYPE_64()(0x02)
	outputDirectMask[7] = VGL_PACK_CL_SHADER_TYPE_64()(0x01)
	outputDirectMask[8] = VGL_PACK_CL_SHADER_TYPE_64()(0x8000)
	outputDirectMask[9] = VGL_PACK_CL_SHADER_TYPE_64()(0x4000)
	outputDirectMask[10] = VGL_PACK_CL_SHADER_TYPE_64()(0x2000)
	outputDirectMask[11] = VGL_PACK_CL_SHADER_TYPE_64()(0x1000)
	outputDirectMask[12] = VGL_PACK_CL_SHADER_TYPE_64()(0x0800)
	outputDirectMask[13] = VGL_PACK_CL_SHADER_TYPE_64()(0x0400)
	outputDirectMask[14] = VGL_PACK_CL_SHADER_TYPE_64()(0x0200)
	outputDirectMask[15] = VGL_PACK_CL_SHADER_TYPE_64()(0x0100)
	outputDirectMask[16] = VGL_PACK_CL_SHADER_TYPE_64()(0x800000)
	outputDirectMask[17] = VGL_PACK_CL_SHADER_TYPE_64()(0x400000)
	outputDirectMask[18] = VGL_PACK_CL_SHADER_TYPE_64()(0x200000)
	outputDirectMask[19] = VGL_PACK_CL_SHADER_TYPE_64()(0x100000)
	outputDirectMask[20] = VGL_PACK_CL_SHADER_TYPE_64()(0x08000000)
	outputDirectMask[21] = VGL_PACK_CL_SHADER_TYPE_64()(0x04000000)
	outputDirectMask[22] = VGL_PACK_CL_SHADER_TYPE_64()(0x02000000)
	outputDirectMask[23] = VGL_PACK_CL_SHADER_TYPE_64()(0x01000000)
	outputDirectMask[24] = VGL_PACK_CL_SHADER_TYPE_64()(0x0800000000)
	outputDirectMask[25] = VGL_PACK_CL_SHADER_TYPE_64()(0x0400000000)
	outputDirectMask[26] = VGL_PACK_CL_SHADER_TYPE_64()(0x0200000000)
	outputDirectMask[27] = VGL_PACK_CL_SHADER_TYPE_64()(0x0100000000)
	outputDirectMask[28] = VGL_PACK_CL_SHADER_TYPE_64()(0x080000000000)
	outputDirectMask[29] = VGL_PACK_CL_SHADER_TYPE_64()(0x040000000000)
	outputDirectMask[30] = VGL_PACK_CL_SHADER_TYPE_64()(0x020000000000)
	outputDirectMask[31] = VGL_PACK_CL_SHADER_TYPE_64()(0x010000000000)
	outputDirectMask[32] = VGL_PACK_CL_SHADER_TYPE_64()(0x08000000000000)
	outputDirectMask[33] = VGL_PACK_CL_SHADER_TYPE_64()(0x04000000000000)
	outputDirectMask[34] = VGL_PACK_CL_SHADER_TYPE_64()(0x02000000000000)
	outputDirectMask[35] = VGL_PACK_CL_SHADER_TYPE_64()(0x01000000000000)
	return outputDirectMask