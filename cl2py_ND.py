"""
    ************************************************************************
    ***                                                                  ***
    ***                Source code generated by cl2py.pl                 ***
    ***                                                                  ***
    ***                        Please do not edit                        ***
    ***                                                                  ***
    ************************************************************************
"""

# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

# TO INFER TYPE TO THE VARIABLE
from typing import Union

#TO WORK WITH MAIN
import numpy as np
import sys

class CL_ND:
    def __init__(self, cl_ctx=None):
        # PYTHON-EXCLUSIVE VARIABLES
        self.cl_ctx: Union[None, vl.opencl_context] = cl_ctx

        # COMMON VARIABLES. self.ocl IS EQUIVALENT TO cl.
        self.ocl: Union[None, vl.VglClContext] = None

        # SE O CONTEXTO OPENCL NÃO FOR DEFINIDO
        # ELE INSTANCIADO E DEFINIDO
        if( self.cl_ctx is None ):
            vl.vglClInit()
            self.ocl = vl.get_ocl()
            self.cl_ctx = vl.get_ocl_context()
        else:
            self.ocl = cl_ctx.get_vglClContext_attributes()

    """
    /** N-dimensional convolution

    SHAPE directive passes a structure with size of each dimension, offsets and number of dimensions. Parameter does not appear in wrapper parameter list. The C expression between parenthesis returns the desired shape of type VglClShape.
    
  */    
    """
    def vglClNdConvolution(self, img_input, img_output, window):


        if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
            print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_input isn't.")
            exit(1)


        if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
            print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_output isn't.")
            exit(1)

        # CREATING OPENCL BUFFER TO VglClShape
        mobj_img_shape = img_input.getVglShape().get_asVglClShape_buffer()

        # EVALUATING IF window IS IN CORRECT TYPE
        if( not isinstance(window, vl.VglStrEl) ):
            print("vglClNdConvolution: Error: window is not a VglClStrEl object. aborting execution.")
            exit()

        # CREATING OPENCL BUFFER TO VglClStrEl
        mobj_window = window.get_asVglClStrEl_buffer()

        vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
        vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

        _program = self.cl_ctx.get_compiled_kernel("../CL_ND/vglClNdConvolution.cl", "vglClNdConvolution")
        _kernel = _program.vglClNdConvolution

        _kernel.set_arg(0, img_input.get_oclPtr())
        _kernel.set_arg(1, img_output.get_oclPtr())
        _kernel.set_arg(2, mobj_img_shape)
        _kernel.set_arg(3, mobj_window)

        # THIS IS A BLOCKING COMMAND. IT EXECUTES THE KERNEL.
        cl.enqueue_nd_range_kernel(self.ocl.commandQueue, _kernel, img_input.get_ipl().shape, None)

        mobj_img_shape = None
        vl.vglSetContext(img_input, vl.VGL_CL_CONTEXT())

        vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

    """
    /** Copy N-dimensional image.

  */    
    """
    def vglClNdCopy(self, img_input, img_output):


        if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
            print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_input isn't.")
            exit(1)


        if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
            print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_output isn't.")
            exit(1)

        vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
        vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

        _program = self.cl_ctx.get_compiled_kernel("../CL_ND/vglClNdCopy.cl", "vglClNdCopy")
        _kernel = _program.vglClNdCopy

        _kernel.set_arg(0, img_input.get_oclPtr())
        _kernel.set_arg(1, img_output.get_oclPtr())

        # THIS IS A BLOCKING COMMAND. IT EXECUTES THE KERNEL.
        cl.enqueue_nd_range_kernel(self.ocl.commandQueue, _kernel, img_input.get_ipl().shape, None)

        vl.vglSetContext(img_input, vl.VGL_CL_CONTEXT())

        vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

    """
    /** N-dimensional dilation

    SHAPE directive passes a structure with size of each dimension, offsets and number of dimensions. Parameter does not appear in wrapper parameter list. The C expression between parenthesis returns the desired shape of type VglClShape.
    
  */    
    """
    def vglClNdDilate(self, img_input, img_output, window):


        if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
            print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_input isn't.")
            exit(1)


        if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
            print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_output isn't.")
            exit(1)

        # CREATING OPENCL BUFFER TO VglClShape
        mobj_img_shape = img_input.getVglShape().get_asVglClShape_buffer()

        # EVALUATING IF window IS IN CORRECT TYPE
        if( not isinstance(window, vl.VglStrEl) ):
            print("vglClNdConvolution: Error: window is not a VglClStrEl object. aborting execution.")
            exit()

        # CREATING OPENCL BUFFER TO VglClStrEl
        mobj_window = window.get_asVglClStrEl_buffer()

        vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
        vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

        _program = self.cl_ctx.get_compiled_kernel("../CL_ND/vglClNdDilate.cl", "vglClNdDilate")
        _kernel = _program.vglClNdDilate

        _kernel.set_arg(0, img_input.get_oclPtr())
        _kernel.set_arg(1, img_output.get_oclPtr())
        _kernel.set_arg(2, mobj_img_shape)
        _kernel.set_arg(3, mobj_window)

        # THIS IS A BLOCKING COMMAND. IT EXECUTES THE KERNEL.
        cl.enqueue_nd_range_kernel(self.ocl.commandQueue, _kernel, img_input.get_ipl().shape, None)

        mobj_img_shape = None
        vl.vglSetContext(img_input, vl.VGL_CL_CONTEXT())

        vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

    """
    /** N-dimensional erosion

    SHAPE directive passes a structure with size of each dimension, offsets and number of dimensions. Parameter does not appear in wrapper parameter list. The C expression between parenthesis returns the desired shape of type VglClShape.
    
  */    
    """
    def vglClNdErode(self, img_input, img_output, window):


        if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
            print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_input isn't.")
            exit(1)


        if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
            print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_output isn't.")
            exit(1)

        # CREATING OPENCL BUFFER TO VglClShape
        mobj_img_shape = img_input.getVglShape().get_asVglClShape_buffer()

        # EVALUATING IF window IS IN CORRECT TYPE
        if( not isinstance(window, vl.VglStrEl) ):
            print("vglClNdConvolution: Error: window is not a VglClStrEl object. aborting execution.")
            exit()

        # CREATING OPENCL BUFFER TO VglClStrEl
        mobj_window = window.get_asVglClStrEl_buffer()

        vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
        vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

        _program = self.cl_ctx.get_compiled_kernel("../CL_ND/vglClNdErode.cl", "vglClNdErode")
        _kernel = _program.vglClNdErode

        _kernel.set_arg(0, img_input.get_oclPtr())
        _kernel.set_arg(1, img_output.get_oclPtr())
        _kernel.set_arg(2, mobj_img_shape)
        _kernel.set_arg(3, mobj_window)

        # THIS IS A BLOCKING COMMAND. IT EXECUTES THE KERNEL.
        cl.enqueue_nd_range_kernel(self.ocl.commandQueue, _kernel, img_input.get_ipl().shape, None)

        mobj_img_shape = None
        vl.vglSetContext(img_input, vl.VGL_CL_CONTEXT())

        vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

    """
    /** Invert N-dimensional image.

  */    
    """
    def vglClNdNot(self, img_input, img_output):


        if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
            print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_input isn't.")
            exit(1)


        if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
            print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_output isn't.")
            exit(1)

        vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
        vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

        _program = self.cl_ctx.get_compiled_kernel("../CL_ND/vglClNdNot.cl", "vglClNdNot")
        _kernel = _program.vglClNdNot

        _kernel.set_arg(0, img_input.get_oclPtr())
        _kernel.set_arg(1, img_output.get_oclPtr())

        # THIS IS A BLOCKING COMMAND. IT EXECUTES THE KERNEL.
        cl.enqueue_nd_range_kernel(self.ocl.commandQueue, _kernel, img_input.get_ipl().shape, None)

        vl.vglSetContext(img_input, vl.VGL_CL_CONTEXT())

        vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

    """
    /** Threshold of img_input by parameter. if the pixel is below thresh,
    the output is 0, else, the output is top. Result is stored in img_output.
  */    
    """
    def vglClNdThreshold(self, img_input, img_output, thresh, top = 255):


        if( not img_input.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
            print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_input isn't.")
            exit(1)


        if( not img_output.clForceAsBuf == vl.IMAGE_ND_ARRAY() ):
            print("vglClNdCopy: Error: this function supports only OpenCL data as buffer and img_output isn't.")
            exit(1)

        vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
        vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())
        # EVALUATING IF thresh IS IN CORRECT TYPE
        if( not isinstance(thresh, np.uint8) ):
            print("vglClConvolution: Warning: thresh not np.uint8! Trying to convert...")
            try:
                thresh = np.uint8(thresh)
            except Exception as e:
                print("vglClConvolution: Error!! Impossible to convert thresh as a np.uint8 object.")
                print(str(e))
                exit()
        # EVALUATING IF top IS IN CORRECT TYPE
        if( not isinstance(top, np.uint8) ):
            print("vglClConvolution: Warning: top not np.uint8! Trying to convert...")
            try:
                top = np.uint8(top)
            except Exception as e:
                print("vglClConvolution: Error!! Impossible to convert top as a np.uint8 object.")
                print(str(e))
                exit()

        _program = self.cl_ctx.get_compiled_kernel("../CL_ND/vglClNdThreshold.cl", "vglClNdThreshold")
        _kernel = _program.vglClNdThreshold

        _kernel.set_arg(0, img_input.get_oclPtr())
        _kernel.set_arg(1, img_output.get_oclPtr())
        _kernel.set_arg(2, thresh)
        _kernel.set_arg(3, top)

        # THIS IS A BLOCKING COMMAND. IT EXECUTES THE KERNEL.
        cl.enqueue_nd_range_kernel(self.ocl.commandQueue, _kernel, img_input.get_ipl().shape, None)

        vl.vglSetContext(img_input, vl.VGL_CL_CONTEXT())

        vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

