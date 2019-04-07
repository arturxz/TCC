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

	ARGV[1]: PRIMARY 2D-IMAGE PATH (COLORED OR GRAYSCALE)
		IT WILL BE USED IN ALL KERNELS

	ARGV[2]: SECONDARY 2D-IMAGE PATH (COLORED OR GRAYSCALE)
		IT WILL BE USED IN THE KERNELS THAT NEED
		TWO INPUT IMAGES TO WORK PROPERLY

	THE RESULT IMAGES WILL BE SAVED AS IMG-[PROCESSNAME].JPG
"""


def salvando2d(img, name):
	# SAVING IMAGE img
	ext = name.split(".")
	ext.reverse()

	#vl.vglClDownload(img)
	vl.vglCheckContext(img, vl.VGL_RAM_CONTEXT())

	if( ext.pop(0).lower() == 'jpg' ):
		if( img.getVglShape().getNChannels() == 4 ):
			vl.rgba_to_rgb(img)
	
	vl.vglSaveImage(name, img)

if __name__ == "__main__":
	
	"""
		CL.IMAGE OBJECTS
	"""

	msg = ""

	vl.vglClInit()

	img_input = vl.VglImage(sys.argv[1], None, vl.VGL_IMAGE_2D_IMAGE())
	vl.vglLoadImage(img_input)
	if( img_input.getVglShape().getNChannels() == 3 ):
		vl.rgb_to_rgba(img_input)
	
	vl.vglClUpload(img_input)
	
	img_input2 = vl.VglImage(sys.argv[2], None, vl.VGL_IMAGE_2D_IMAGE())
	vl.vglLoadImage(img_input2)
	if( img_input2.getVglShape().getNChannels() == 3 ):
		vl.rgb_to_rgba(img_input2)

	img_output = vl.create_blank_image_as(img_input)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
	vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())

	convolution_window_2d = np.ones((5,5), np.float32) * (1/25)

	morph_window_2d = np.ones((3,3), np.uint8) * 255
	morph_window_2d[0,0] = 0 
	morph_window_2d[0,2] = 0
	morph_window_2d[2,0] = 0
	morph_window_2d[2,2] = 0

	inicio = t.time()
	vglClBlurSq3(img_input, img_output)
	fim = t.time()
	salvando2d(img_output, "img-vglClBlurSq3.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClBlurSq3:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClConvolution(img_input, img_output, convolution_window_2d, np.uint32(5), np.uint32(5))
	fim = t.time()
	salvando2d(img_output, "img-vglClConvolution.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClConvolution:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClCopy(img_input, img_output)
	fim = t.time()
	salvando2d(img_output, "img-vglClCopy.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClCopy:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClDilate(img_input, img_output, morph_window_2d, np.uint32(3), np.uint32(3))
	fim = t.time()
	salvando2d(img_output, "img-vglClDilate.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClDilate:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClErode(img_input, img_output, morph_window_2d, np.uint32(3), np.uint32(3))
	fim = t.time()
	salvando2d(img_output, "img-vglClErode.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClErode:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClInvert(img_input, img_output)
	fim = t.time()
	salvando2d(img_output, "img-vglClInvert.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClInvert:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClMax(img_input, img_input2, img_output)
	fim = t.time()
	salvando2d(img_output, "img-vglClMax.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClMax:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClMin(img_input, img_input2, img_output)
	fim = t.time()
	salvando2d(img_output, "img-vglClMin.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClMin:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClSub(img_input, img_input2, img_output)
	fim = t.time()
	salvando2d(img_output, "img-vglClSub.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClSub:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClSum(img_input, img_input2, img_output)
	fim = t.time()
	salvando2d(img_output, "img-vglClSum.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClSum:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClSwapRgb(img_input, img_output)
	fim = t.time()
	salvando2d(img_output, "img-vglClSwapRgb.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClSwapRgb:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClThreshold(img_input, img_output, np.float32(0.5), np.float32(0.9))
	fim = t.time()
	salvando2d(img_output, "img-vglClThreshold.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClThreshold:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	print("-------------------------------------------------------------")
	print(msg)
	print("-------------------------------------------------------------")

	img_input = None
	img_input2 = None
	
	img_output = None
	
	convolution_window_2d = None	
	morph_window_2d = None
