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
	img_in_path = sys.argv[1]
	nSteps		= int(sys.argv[2])
	img_out_path= sys.argv[3]

	print("in path", img_in_path)
	print("steps", nSteps)
	print("out path", img_out_path)

	msg = ""

	vl.vglClInit()

	img_input_3d = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_3D_IMAGE())
	vl.vglLoadImage(img_input_3d)
	vl.vglClUpload(img_input_3d)

	img_output_3d = vl.create_blank_image_as(img_input_3d)
	img_output_3d.set_oclPtr( vl.get_similar_oclPtr_object(img_input_3d) )
	
	vl.vglAddContext(img_output_3d, vl.VGL_CL_CONTEXT())

	convolution_window_3d_3x3x3 = np.ones((3,3,3), np.float32) * (1/27)
	convolution_window_3d_5x5x5 = np.ones((5,5,5), np.float32) * (1/125)
	
	vglCl3dBlurSq3(img_input_3d, img_output_3d)
	media = 0.0
	for i in range(0, 5):
		p = 0
		inicio = t.time()
		while(p < nSteps):
			vglCl3dBlurSq3(img_input_3d, img_output_3d)
			p = p + 1
		fim = t.time()
		media = media + (fim-inicio)
	
	vl.vglSaveImage(img_out_path+"3d-vglCl3dBlurSq3.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dBlurSq3:\t\t" +str( round( (media / 5), 9 ) ) +"s\n"	

	vglCl3dConvolution(img_input_3d, img_output_3d, convolution_window_3d_3x3x3, np.uint32(3), np.uint32(3), np.uint32(3))
	media = 0.0
	for i in range(0, 5):
		p = 0
		inicio = t.time()
		while(p < nSteps):
			vglCl3dConvolution(img_input_3d, img_output_3d, convolution_window_3d_3x3x3, np.uint32(3), np.uint32(3), np.uint32(3))
			p = p + 1
		fim = t.time()
		media = media + (fim-inicio)
	
	vl.vglSaveImage(img_out_path+"3d-vglCl3dConvolution-3.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dConvolution (3x3x3):\t" +str( round( (media / 5), 9 ) ) +"s\n"	

	vglCl3dConvolution(img_input_3d, img_output_3d, convolution_window_3d_5x5x5, np.uint32(5), np.uint32(5), np.uint32(5))
	media = 0.0
	for i in range(0, 5):
		p = 0
		inicio = t.time()
		while(p < nSteps):
			vglCl3dConvolution(img_input_3d, img_output_3d, convolution_window_3d_5x5x5, np.uint32(5), np.uint32(5), np.uint32(5))
			p = p + 1
		fim = t.time()
		media = media + (fim-inicio)
	
	vl.vglSaveImage(img_out_path+"3d-vglCl3dConvolution-5.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dConvolution (5x5x5):\t" +str( round( (media / 5), 9 ) ) +"s\n"	

	vglCl3dNot(img_input_3d, img_output_3d)
	media = 0.0
	for i in range(0, 5):
		p = 0
		inicio = t.time()
		while(p < nSteps):
			vglCl3dNot(img_input_3d, img_output_3d)
			p = p + 1
		fim = t.time()
		media = media + (fim-inicio)
	
	vl.vglSaveImage(img_out_path+"3d-vglCl3dNot.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dNot:\t\t\t" +str( round( (media / 5), 9 ) ) +"s\n"	

	vglCl3dThreshold(img_input_3d, img_output_3d, np.float32(0.4), np.float32(.8))
	media = 0.0
	for i in range(0, 5):
		p = 0
		inicio = t.time()
		while(p < nSteps):
			vglCl3dThreshold(img_input_3d, img_output_3d, np.float32(0.4), np.float32(.8))
			p = p + 1
		fim = t.time()
		media = media + (fim-inicio)
	
	vl.vglSaveImage(img_out_path+"3d-vglCl3dThreshold.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dThreshold:\t\t" +str( round( (media / 5), 9 ) ) +"s\n"

	vglCl3dCopy(img_input_3d, img_output_3d)
	media = 0.0
	for i in range(0, 5):
		p = 0
		inicio = t.time()
		while(p < nSteps):
			vglCl3dCopy(img_input_3d, img_output_3d)
			p = p + 1
		fim = t.time()
		media = media + (fim-inicio)
	
	vl.vglSaveImage(img_out_path+"3d-vglCl3dCopy.tif", img_output_3d)
	msg = msg + "Tempo de execução do método vglCl3dCopy:\t\t" +str( round( (media / 5), 9 ) ) +"s\n"	

	print("-------------------------------------------------------------")
	print(msg)
	print("-------------------------------------------------------------")


	img_input_3d = None
	img_input2_3d = None

	img_output_3d = None

	convolution_window_3d = None	
	morph_window_3d = None
