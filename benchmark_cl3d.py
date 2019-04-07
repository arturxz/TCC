#!/usr/bin/env python3

# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

# TO WORK WITH MAIN
import numpy as np

# IMPORTING METHODS
from cl2py_shaders import * 

import time as t
import sys

"""
	THIS BENCHMARK TOOL EXPECTS 2 ARGUMENTS:

	ARGV[1]: PRIMARY 3D-IMAGE PATH (PREFERENCE TIFF FILE)
		IT WILL BE USED IN ALL KERNELS

	ARGV[2]: SECONDARY 3D-IMAGE PATH (PREFERENCE AS TIFF FILE)
		IT WILL BE USED IN THE KERNELS THAT NEED
		TWO INPUT IMAGES TO WORK PROPERLY

	THE RESULT IMAGES WILL BE SAVED AS 3d-[PROCESSNAME].TIFF
"""

if __name__ == "__main__":
	
	"""
		CL.IMAGE OBJECTS
	"""

	msg = ""

	vl.vglClInit()

	img_input_3d = vl.VglImage(sys.argv[1], None, vl.VGL_IMAGE_3D_IMAGE())
	vl.vglLoadImage(img_input_3d)
	vl.vglClUpload(img_input_3d)

	img_input2_3d = vl.VglImage(sys.argv[2], None, vl.VGL_IMAGE_3D_IMAGE())
	vl.vglLoadImage(img_input2_3d)

	img_output_3d = vl.create_blank_image_as(img_input_3d)
	img_output_3d.set_oclPtr( vl.get_similar_oclPtr_object(img_input_3d) )
	vl.vglAddContext(img_output_3d, vl.VGL_CL_CONTEXT())

	convolution_window_3d = np.ones((5,5,5), np.float32) * (1/125)

	morph_window_3d = np.zeros((3,3,3), np.uint8)
	morph_window_3d[0,1,1] = 255
	morph_window_3d[1,0,1] = 255
	morph_window_3d[1,1,0] = 255
	morph_window_3d[1,1,1] = 255
	morph_window_3d[1,1,2] = 255
	morph_window_3d[1,2,1] = 255
	morph_window_3d[2,1,1] = 255

	inicio = t.time()
	vglCl3dBlurSq3(img_input_3d, img_output_3d)
	fim = t.time()
	vl.vglSaveImage("3d-vglCl3dBlurSq3.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dBlurSq3:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dConvolution(img_input_3d, img_output_3d, convolution_window_3d, np.uint32(5), np.uint32(5), np.uint32(5))
	fim = t.time()
	vl.vglSaveImage("3d-vglCl3dConvolution.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dConvolution:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dCopy(img_input_3d, img_output_3d)
	fim = t.time()
	vl.vglSaveImage("3d-vglCl3dCopy.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dCopy:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dDilate(img_input_3d, img_output_3d, morph_window_3d, np.uint32(3), np.uint32(3), np.uint32(3))
	fim = t.time()
	vl.vglSaveImage("3d-vglCl3dDilate.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dDilate:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dErode(img_input_3d, img_output_3d, morph_window_3d, np.uint32(3), np.uint32(3), np.uint32(3))
	fim = t.time()
	vl.vglSaveImage("3d-vglCl3dErode.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dErode:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dNot(img_input_3d, img_output_3d)
	fim = t.time()
	vl.vglSaveImage("3d-vglCl3dNot.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dNot:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dMax(img_input_3d, img_input2_3d, img_output_3d)
	fim = t.time()
	vl.vglSaveImage("3d-vglCl3dMax.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dMax:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dMin(img_input_3d, img_input2_3d, img_output_3d)
	fim = t.time()
	vl.vglSaveImage("3d-vglCl3dMin.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dMin:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dSub(img_input_3d, img_input2_3d, img_output_3d)
	fim = t.time()
	vl.vglSaveImage("3d-vglCl3dSub.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dSub:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dSum(img_input_3d, img_input2_3d, img_output_3d)
	fim = t.time()
	vl.vglSaveImage("3d-vglCl3dSum.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dSum:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dThreshold(img_input_3d, img_output_3d, np.float32(0.4), np.float32(.8))
	fim = t.time()
	vl.vglSaveImage("3d-vglCl3dThreshold.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dThreshold:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	print("-------------------------------------------------------------")
	print(msg)
	print("-------------------------------------------------------------")


	img_input_3d = None
	img_input2_3d = None

	img_output_3d = None

	convolution_window_3d = None	
	morph_window_3d = None
