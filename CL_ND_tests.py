#!/bin/python

# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

# TO WORK WITH MAIN
import numpy as np

# IMPORTING METHODS
from cl2py_ND import * 

import time as t

"""
	HERE FOLLOWS THE KERNEL CALLS
"""
if __name__ == "__main__":
	
	vl.vglClInit()

	msg = ""

	# INPUT IMAGE
	img_input = vl.VglImage("img-1.jpg", None, vl.VGL_IMAGE_2D_IMAGE(), vl.IMAGE_ND_ARRAY())
	vl.vglLoadImage(img_input)
	vl.vglClUpload(img_input)

	# OUTPUT IMAGE
	img_output = vl.create_blank_image_as(img_input)
	img_output.set_oclPtr( vl.get_similar_oclPtr_object(img_input) )
	vl.vglAddContext(img_output, vl.VGL_CL_CONTEXT())

	# STRUCTURANT ELEMENT
	window = vl.VglStrEl()
	window.constructorFromTypeNdim(vl.VGL_STREL_CROSS(), 2)

	inicio = t.time()
	vglClNdCopy(img_input, img_output)
	fim = t.time()
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdCopy.jpg", img_output)
	msg = msg + "Tempo de execução do método vglClNdCopy:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"

	inicio = t.time()
	vglClNdConvolution(img_input, img_output, window)
	fim = t.time()
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdConvolution.jpg", img_output)
	msg = msg + "Tempo de execução do método vglClNdConvolution:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"

	inicio = t.time()
	vglClNdDilate(img_input, img_output, window)
	fim = t.time()
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdDilate.jpg", img_output)
	msg = msg + "Tempo de execução do método vglClNdDilate:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"

	inicio = t.time()
	vglClNdErode(img_input, img_output, window)
	fim = t.time()
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdErode.jpg", img_output)
	msg = msg + "Tempo de execução do método vglClNdErode:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"

	inicio = t.time()
	vglClNdNot(img_input, img_output)
	fim = t.time()
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdNot.jpg", img_output)
	msg = msg + "Tempo de execução do método vglClNdNot:\t\t" +str( round( (fim-inicio), 9 ) ) +"s\n"

	inicio = t.time()
	vglClNdThreshold(img_input, img_output, np.uint8(120), np.uint8(190))
	fim = t.time()
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdThreshold.jpg", img_output)
	msg = msg + "Tempo de execução do método vglClNdThreshold:\t" +str( round( (fim-inicio), 9 ) ) +"s\n"

	print("-------------------------------------------------------------")
	print(msg)
	print("-------------------------------------------------------------")

	wrp = None
	img_input = None
	img_output = None
	window = None

