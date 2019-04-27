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

	img_in_path = sys.argv[1]
	nSteps		= int(sys.argv[2])
	img_out_path= sys.argv[3]

	msg = ""

	vl.vglClInit()

	img_input = vl.VglImage(img_in_path, None, vl.VGL_IMAGE_2D_IMAGE())
	vl.vglLoadImage(img_input)
	if( img_input.getVglShape().getNChannels() == 3 ):
		vl.rgb_to_rgba(img_input)
	
	vl.vglClUpload(img_input)
	
	img_output = vl.create_blank_image_as(img_input)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
	vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())
	
	convolution_window_2d_3x3 = np.array((	(1/16, 2/16, 1/16),
											(2/16, 4/16, 2/16),
											(1/16, 2/16, 1/16) ), np.float32)
	convolution_window_2d_5x5 = np.array((	(1/256, 4/256,  6/256,  4/256,  1/256),
											(4/256, 16/256, 24/256, 16/256, 4/256),
											(6/256, 24/256, 36/256, 24/256, 6/256),
											(4/256, 16/256, 24/256, 16/256, 4/256),
											(1/256, 4/256,  6/256,  4/256,  1/256) ), np.float32)
	vglClBlurSq3(img_input, img_output)
	media = 0.0
	for i in range(0, 5):
		p = 0
		inicio = t.time()
		while(p < nSteps):
			vglClBlurSq3(img_input, img_output)
			p = p + 1
		fim = t.time()
		media = media + (fim-inicio)

	salvando2d(img_output, img_out_path+"img-vglClBlurSq3.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClBlurSq3:\t\t" +str( round( ( media / 5 ), 9 ) ) +"s\n"	

	vglClConvolution(img_input, img_output, convolution_window_2d_3x3, np.uint32(5), np.uint32(5))
	media = 0.0
	for i in range(0, 5):
		p = 0
		inicio = t.time()
		while(p < nSteps):
			vglClConvolution(img_input, img_output, convolution_window_2d_3x3, np.uint32(5), np.uint32(5))
			p = p + 1
		fim = t.time()
		media = media + (fim-inicio)

	salvando2d(img_output, img_out_path+"img-vglClConvolution.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClConvolution 3x3:\t" +str( round( (media / 5), 9 ) ) +"s\n"

	vglClConvolution(img_input, img_output, convolution_window_2d_5x5, np.uint32(5), np.uint32(5))
	media = 0.0
	for i in range(0, 5):
		p = 0
		inicio = t.time()
		while(p < nSteps):
			vglClConvolution(img_input, img_output, convolution_window_2d_5x5, np.uint32(5), np.uint32(5))
			p = p + 1
		fim = t.time()
		media = media + (fim-inicio)
	
	salvando2d(img_output, img_out_path+"img-vglClConvolution.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClConvolution 5x5:\t" +str( round( (media / 5), 9 ) ) +"s\n"

	vglClInvert(img_input, img_output)
	media = 0.0
	for i in range(0, 5):
		p = 0
		inicio = t.time()
		while(p < nSteps):
			vglClInvert(img_input, img_output)
			p = p + 1
		fim = t.time()
		media = media + (fim-inicio)

	salvando2d(img_output, img_out_path+"img-vglClInvert.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClInvert:\t\t" +str( round( (media / 5), 9 ) ) +"s\n"	

	vglClThreshold(img_input, img_output, np.float32(0.5))
	media = 0.0
	for i in range(0, 5):
		p = 0
		inicio = t.time()
		while(p < nSteps):
			vglClThreshold(img_input, img_output, np.float32(0.5))
			p = p + 1
		fim = t.time()
		media = media + (fim-inicio)
	
	salvando2d(img_output, img_out_path+"img-vglClThreshold.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClThreshold:\t\t" +str( round( (media / 5), 9 ) ) +"s\n"	

	vglClCopy(img_input, img_output)
	media = 0.0
	for i in range(0, 5):
		p = 0
		inicio = t.time()
		while(p < nSteps):
			vglClCopy(img_input, img_output)
			p = p + 1
		fim = t.time()
		media = media + (fim-inicio)

	salvando2d(img_output, img_out_path+"img-vglClCopy.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClCopy:\t\t\t" +str( round( (media / 5), 9 ) ) +"s"	

	print("-------------------------------------------------------------")
	print(msg)
	print("-------------------------------------------------------------")

	img_input = None
	img_output = None
	
	convolution_window_2d_5x5 = None
	convolution_window_2d_3x3 = None