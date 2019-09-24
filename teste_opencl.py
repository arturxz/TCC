#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyopencl as cl
import numpy as np

def device_type_as_string( dt ):
    if( dt == cl.device_type.ACCELERATOR ):
        return "ACCELERATOR"
    if( dt == cl.device_type.ALL ):
        return "ALL"
    if( dt == cl.device_type.CPU ):
        return "CPU"
    if( dt == cl.device_type.CUSTOM ):
        return "CUSTOM"
    if( dt == cl.device_type.DEFAULT ):
        return "DEFAULT"
    if( dt == cl.device_type.GPU ):
        return "GPU"
    if( dt == cl.device_type.mro ):
        return "mro"

if __name__ == "__main__":

    print( len( cl.get_platforms() ), "platforms found:" )
    for platform in cl.get_platforms():
        print( "   Platform:", platform.name )
        print( "  ", len( platform.get_devices() ), "devices found:" )
        
        for device in platform.get_devices():
            print( "      Device name:", device.name )
            print( "      Device type:", device_type_as_string( device.type ) )
            print( "      ------------------------------------------" )

            ctx  = cl.Context( [ device ] )
            fila = cl.CommandQueue( ctx )

            # PREPARING TO RUN StructSizes
            host = np.zeros( 11, np.uint32 )
