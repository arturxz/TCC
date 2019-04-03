#!/bin/python

# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

# TO WORK WITH MAIN
import numpy as np

# IMPORTING METHODS
from cl2py_shaders import * 

import time as t

"""
	HERE FOLLOWS THE KERNEL CALLS
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

	img_input = vl.VglImage("img-1.jpg", None, vl.VGL_IMAGE_2D_IMAGE())
	vl.vglLoadImage(img_input)
	if( img_input.getVglShape().getNChannels() == 3 ):
		vl.rgb_to_rgba(img_input)
	
	vl.vglClUpload(img_input)
	
	img_input2 = vl.VglImage("img-2.jpg", None, vl.VGL_IMAGE_2D_IMAGE())
	vl.vglLoadImage(img_input2)
	if( img_input2.getVglShape().getNChannels() == 3 ):
		vl.rgb_to_rgba(img_input2)

	img_input_3d = vl.VglImage("3d-1.tif", None, vl.VGL_IMAGE_3D_IMAGE())
	vl.vglLoadImage(img_input_3d)
	vl.vglClUpload(img_input_3d)

	img_input2_3d = vl.VglImage("3d-2.tif", None, vl.VGL_IMAGE_3D_IMAGE())
	vl.vglLoadImage(img_input2_3d)

	img_output = vl.create_blank_image_as(img_input)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
	vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())

	img_output_3d = vl.create_blank_image_as(img_input_3d)
	img_output_3d.set_oclPtr( vl.get_similar_oclPtr_object(img_input_3d) )
	vl.vglAddContext(img_output_3d, vl.VGL_CL_CONTEXT())

	convolution_window_2d = np.ones((5,5), np.float32) * (1/25)
	convolution_window_3d = np.ones((5,5,5), np.float32) * (1/125)

	morph_window_2d = np.ones((3,3), np.uint8) * 255
	morph_window_2d[0,0] = 0 
	morph_window_2d[0,2] = 0
	morph_window_2d[2,0] = 0
	morph_window_2d[2,2] = 0

	morph_window_3d = np.zeros((3,3,3), np.uint8)
	morph_window_3d[0,1,1] = 255
	morph_window_3d[1,0,1] = 255
	morph_window_3d[1,1,0] = 255
	morph_window_3d[1,1,1] = 255
	morph_window_3d[1,1,2] = 255
	morph_window_3d[1,2,1] = 255
	morph_window_3d[2,1,1] = 255

	inicio = t.time()
	vglClBlurSq3(img_input, img_output)
	fim = t.time()
	salvando2d(img_output, "yamamoto-vglClBlurSq3.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClBlurSq3:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClConvolution(img_input, img_output, convolution_window_2d, np.uint32(5), np.uint32(5))
	fim = t.time()
	salvando2d(img_output, "yamamoto-vglClConvolution.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClConvolution:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClCopy(img_input, img_output)
	fim = t.time()
	salvando2d(img_output, "yamamoto-vglClCopy.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClCopy:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClDilate(img_input, img_output, morph_window_2d, np.uint32(3), np.uint32(3))
	fim = t.time()
	salvando2d(img_output, "yamamoto-vglClDilate.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClDilate:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClErode(img_input, img_output, morph_window_2d, np.uint32(3), np.uint32(3))
	fim = t.time()
	salvando2d(img_output, "yamamoto-vglClErode.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClErode:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClInvert(img_input, img_output)
	fim = t.time()
	salvando2d(img_output, "yamamoto-vglClInvert.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClInvert:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClMax(img_input, img_input2, img_output)
	fim = t.time()
	salvando2d(img_output, "yamamoto-vglClMax.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClMax:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClMin(img_input, img_input2, img_output)
	fim = t.time()
	salvando2d(img_output, "yamamoto-vglClMin.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClMin:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClSub(img_input, img_input2, img_output)
	fim = t.time()
	salvando2d(img_output, "yamamoto-vglClSub.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClSub:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClSum(img_input, img_input2, img_output)
	fim = t.time()
	salvando2d(img_output, "yamamoto-vglClSum.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClSum:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClSwapRgb(img_input, img_output)
	fim = t.time()
	salvando2d(img_output, "yamamoto-vglClSwapRgb.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClSwapRgb:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglClThreshold(img_input, img_output, np.float32(0.5), np.float32(0.9))
	fim = t.time()
	salvando2d(img_output, "yamamoto-vglClThreshold.jpg")
	vl.rgb_to_rgba(img_output)
	msg = msg + "Tempo de execução do método vglClThreshold:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dBlurSq3(img_input_3d, img_output_3d)
	fim = t.time()
	salvando2d(img_output_3d, "3d-vglCl3dBlurSq3.tif")
	msg = msg + "Tempo de execução do método vglCl3dBlurSq3:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dConvolution(img_input_3d, img_output_3d, convolution_window_3d, np.uint32(5), np.uint32(5), np.uint32(5))
	fim = t.time()
	salvando2d(img_output_3d, "3d-vglCl3dConvolution.tif")
	msg = msg + "Tempo de execução do método vglCl3dConvolution:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dCopy(img_input_3d, img_output_3d)
	fim = t.time()
	salvando2d(img_output_3d, "3d-vglCl3dCopy.tif")
	msg = msg + "Tempo de execução do método vglCl3dCopy:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dDilate(img_input_3d, img_output_3d, morph_window_3d, np.uint32(3), np.uint32(3), np.uint32(3))
	fim = t.time()
	salvando2d(img_output_3d, "3d-vglCl3dDilate.tif")
	msg = msg + "Tempo de execução do método vglCl3dDilate:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dErode(img_input_3d, img_output_3d, morph_window_3d, np.uint32(3), np.uint32(3), np.uint32(3))
	fim = t.time()
	salvando2d(img_output_3d, "3d-vglCl3dErode.tif")
	msg = msg + "Tempo de execução do método vglCl3dErode:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dNot(img_input_3d, img_output_3d)
	fim = t.time()
	salvando2d(img_output_3d, "3d-vglCl3dNot.tif")
	msg = msg + "Tempo de execução do método vglCl3dNot:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dMax(img_input_3d, img_input2_3d, img_output_3d)
	fim = t.time()
	salvando2d(img_output_3d, "3d-vglCl3dMax.tif")
	msg = msg + "Tempo de execução do método vglCl3dMax:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dMin(img_input_3d, img_input2_3d, img_output_3d)
	fim = t.time()
	salvando2d(img_output_3d, "3d-vglCl3dMin.tif")
	msg = msg + "Tempo de execução do método vglCl3dMin:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dSub(img_input_3d, img_input2_3d, img_output_3d)
	fim = t.time()
	salvando2d(img_output_3d, "3d-vglCl3dSub.tif")
	msg = msg + "Tempo de execução do método vglCl3dSub:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dSum(img_input_3d, img_input2_3d, img_output_3d)
	fim = t.time()
	salvando2d(img_output_3d, "3d-vglCl3dSum.tif")
	msg = msg + "Tempo de execução do método vglCl3dSum:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	inicio = t.time()
	vglCl3dThreshold(img_input_3d, img_output_3d, np.float32(0.4), np.float32(.8))
	fim = t.time()
	salvando2d(img_output_3d, "3d-vglCl3dThreshold.tif")
	msg = msg + "Tempo de execução do método vglCl3dThreshold:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"	

	print("-------------------------------------------------------------")
	print(msg)
	print("-------------------------------------------------------------")


	img_input = None
	img_input_3d = None

	img_input2 = None
	img_input2_3d = None
	
	img_output = None
	img_output_3d = None
	
	convolution_window_2d = None
	convolution_window_3d = None
	
	morph_window_2d = None
	morph_window_3d = None
