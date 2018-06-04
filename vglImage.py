from skimage import io
import matplotlib.pyplot as mp
import pyopencl as cl
import numpy as np
import sys

# VISIONGL IMPORTS
from vglShape import *
from vglStrEl import *
import vglConst as vc

class vglImage(object):
    def __init__(self, imgPath):
        self.img_host = None
        self.img_device = None
        self.sync = False
        
        try:
            self.img_host = io.imread(imgPath)
            setImgData()
            print("vglImage Object Created")
        except FileNotFoundError as fnf:
            print("Image wasn't found.")    
        except Exception as e:
            print("Unrecognized error.")
    
    def setImgData(self):
        print("Setting image Data")
        self.imgWidth = img.shape[1]
        self.imgHeight = img.shape[0]

	self.img = io.imread(imgpath)
	self.img_dtype = self.img.dtype
	self.img_ndim = self.img.ndim
	
	self.img_shape = (self.imgWidth, self.imgHeight)
	self.img_origin = (0, 0, 0)
	self.img_region = (self.imgHeight, self.imgWidth, 1)

    def vglUpload(self, queue):
        """
            ctx is the opencl context
            and queue is the queue in that ctx
        """
        if( self.img.dtype == np.uint8 ):
            self.img_dtype_cl = cl.channel_type.UNORM_INT8
	elif( self.img.dtype == np.uint16 ):
		self.img_dtype_cl = cl.channel_type.UNORM_INT16

	# GETTING THE DIMENSIONS OF THE IMAGE
	if( self.img_ndim == 1 ):
		# WHAT TO DO IF THE IMAGE IS 1-DIMENSIONAL
		self.img_channel_order_cl = cl.channel_order.LUMINANCE
	elif( self.img_ndim == 2 ):
		# IF THE IMAGE IS 2-DIMENSIONAL, THEN IT IS A SHADES OF GRAY IMAGE
		# AND THE IMAGE TYPE IS LUMINANCE
		self.img_channel_order_cl = cl.channel_order.LUMINANCE
		self.img_nchannels = 1
	elif( self.img_ndim == 3 ):
		# IF THE IMAGE ARRAY IS 3-DIMENSIONAL, THEN IT HAS MORE THAN 1 COLOR CHANNEL

		if( self.img[0,0,:].size == 2 ):
			# THEN IT CAN BE ANY 2-CHANNEL IMAGE.
			# DON'T HAVE ACCES TO ANY IMAGE LIKE THAT YET.
			self.img_nchannels = 2

		if( self.img[0,0,:].size == 3 ):
			# THEN IT CAN BE ANY 3-CHANNEL IMAGE. IS NEEDED TO ADD THE 4TH CHANNEL TO IT
			self.img_nchannels = 3

			"""
				DON'T KNOW YET HOW TO DISCOVER IF IT IS RGB, RBG, BGR, GBR, HSV OR SOME OTHER TYPE.
				FOR NOW, THIS PROGRAM WAS JUST TESTED WITH RGB AND SHOULD WORK WITH ITS 3-CHANNEL VARIANTS.
				HERE IS JUST ADDED AN ALPHA CHANNEL FOR THE IMAGE TO BE IN RGBA (BGRA, RBGA, [...], FORMAT)
			"""

			# TURNING INTO RGBA IMAGE
			self.img_aux = np.empty((self.img.shape[0],self.img.shape[1],4), self.img_dtype)
			self.img_aux[:,:,0] = self.img[:,:,0]
			self.img_aux[:,:,1] = self.img[:,:,1]
			self.img_aux[:,:,2] = self.img[:,:,2]
			self.img_aux[:,:,3] = 255

			self.img = self.img_aux
			self.img_channel_order_cl = cl.channel_order.RGBA

	elif( self.img_ndim == 4 ):
		# THEN IT COULD BE ANY IMAGE WITH 4 COLOR CHANNELS
		# DON'T NEED TO DO ENYTHING WITH THE IMAGE
		# SO, JUST MAKING THE IMAGE FORMAT OBJECT
		self.img_channel_order_cl = cl.channel_order.RGBA
		
	# SETTING THE OPENCL IMAGE OBJECTS, WITHOUT THE COPY
	self.mf = cl.mem_flags
	self.imgFormat = cl.ImageFormat(self.img_channel_order_cl, self.img_dtype_cl)
	self.img_in_cl = cl.Image(ctx, self.mf.READ_ONLY, self.imgFormat, self.img_shape)
        self.img_out_cl = cl.Image(ctx, self.mf.WRITE_ONLY, self.imgFormat, self.img_shape)

	# COPYING NDARRAY IMAGE TO OPENCL IMAGE OBJECT
	cl.enqueue_copy(queue, self.img_in_cl, self.img.tobytes(), origin=self.img_origin, region=self.img_region, is_blocking=True)
	self.img_device = self.img_in_cl

    def vglDownload(self, queue):
		# CREATING BUFFER TO GET IMAGE FROM DEVICE
		if( self.img_nchannels == 1 ):
			self.buffer = np.zeros(self.img_shape[0]*self.img_shape[1], self.img_dtype)
		elif( self.img_nchannels == 3 or self.img_nchannels == 4 ):
			self.buffer = np.zeros(self.img_shape[0]*self.img_shape[1]*4, self.img_dtype)

		cl.enqueue_copy(queue, self.buffer, self.img_out_cl, origin=self.img_origin, region=self.img_region, is_blocking=True)
		
		# TURNING BUFFER INTO A NDARRAY WITH SHAPE SENDED TO THE DEVICE 
		# AND IMG_OUT NDARRAY, THAT HAS THE CORRECT SHAPE TO SAVE IMAGE (AS THE READED IMAGE IS RGB)
		if( self.img_nchannels == 1 ):
			self.buffer = np.frombuffer( self.buffer, self.img_dtype ).reshape( self.img_shape[1], self.img_shape[0] )
		
		elif( self.img_nchannels == 3 or self.img_nchannels == 4 ):
			self.buffer = np.frombuffer( self.buffer, self.img_dtype ).reshape( self.img_shape[1], self.img_shape[0], 4 )
			self.img_out = np.empty( (self.buffer.shape[0],self.buffer.shape[1],self.img_nchannels), dtype=self.img_dtype )

		# PREPARING TO SAVE THE IMAGE
		if( self.img_nchannels == 1 ):
			# GRAY-SHADES IMAGE
			self.img_out = self.buffer

		elif( self.img_nchannels == 2 ):
			# TWO COLOR CHANNEL IMAGE
			self.img_out = self.buffer
		
		elif( self.img_nchannels == 3 ):
			# THEN WAS ADDED A ALPHA CHANNEL TO IT.
			# NOW, THIS ALPHA CHANNEL WILL BE REMOVED AND THEN THE IMAGE WILL BE SAVED.
			self.img_out[:,:,0] = self.buffer[:,:,0]
			self.img_out[:,:,1] = self.buffer[:,:,1]
			self.img_out[:,:,2] = self.buffer[:,:,2]

		elif( self.img_nchannels == 4 ):
			# THEN ITS WAS A RGBA IMAGE, AND DO NOT NEED ALTERATIONS IN IT.
			self.img_out = self.buffer
		
		self.img_ram = self.img_out

    def vglSaveImage(self, outputpath):
        # SAVING IMAGE (USING PIL PLUGIN)
        io.imsave(outputpath, self.img_ram)