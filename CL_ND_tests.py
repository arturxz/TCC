#!/bin/python

# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

# TO WORK WITH MAIN
import numpy as np

# IMPORTING METHODS
from cl2py_ND import * 

"""
	HERE FOLLOWS THE KERNEL CALLS
"""
if __name__ == "__main__":
	
	vl.vglClInit()

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

	vglClNdCopy(img_input, img_output)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdCopy.jpg", img_output)

	vglClNdConvolution(img_input, img_output, window)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdConvolution.jpg", img_output)
	
	vglClNdDilate(img_input, img_output, window)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdDilate.jpg", img_output)

	vglClNdErode(img_input, img_output, window)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdErode.jpg", img_output)

	vglClNdNot(img_input, img_output)
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdNot.jpg", img_output)

	vglClNdThreshold(img_input, img_output, np.uint8(120), np.uint8(190))
	vl.vglCheckContext(img_output, vl.VGL_RAM_CONTEXT())
	vl.vglSaveImage("yamamoto-vglNdThreshold.jpg", img_output)

	wrp = None
	img_input = None
	img_output = None
	window = None

