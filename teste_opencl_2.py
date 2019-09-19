#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyopencl as cl
import numpy as np
import vgl_lib as vl

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

"""
    RETURNS THE DEVICE WITH MORE CAPACITY
    OF WORK ITEMS STORAGE.

    @input: PREFERED DEVICE TYPE
            THE FUNCTION WILL RETURN THE DEVICE
            WITH MORE WORK ITEMS STORAGE OF THE
            TYPE INFERED HERE. THE TYPE MUST BE
            ONE LISTED IN cl.device_type

            IF THERE ISN'T THE REQUIRED TYPE DEVICE,
            THE FUNCTION TRY TO LOCATE A GPU DEVICE.
            IF THERE IS NO GPU DEVICE, IT REURNS THE
            ONE MORE CAPABLE OG WORK ITEMS.
"""
def get_device( device_type=cl.device_type.GPU ):
    dv_type = None
    dv_not_type = None
    
    for platform in cl.get_platforms():
        max_wis_type = np.iinfo( np.int32 ).min
        max_wis_not_type = np.iinfo( np.int32 ).min
        
        for device in platform.get_devices():
            atual_wis = 1
            for n in device.max_work_item_sizes:
                atual_wis = atual_wis * n

            if( device.type == device_type ):
                if( max_wis_type < atual_wis ):
                    max_wis_type = atual_wis
                    dv_type = device
            else:
                if( max_wis_not_type < atual_wis ):
                    max_wis_not_type = atual_wis
                    dv_not_type = device

    # RETURNING BEST OPTION
    if( dv_type == None ):
        return dv_not_type
    else:
        return dv_type


if __name__ == "__main__":
    """
    print( len( cl.get_platforms() ), "platforms found:" )
    for platform in cl.get_platforms():
        print( "   Platform:", platform.name )
        print( "   Platform vendor:", platform.vendor )
        print( "  ", len( platform.get_devices() ), "devices found:" )
        
        for device in platform.get_devices():
            print( "      Device name:", device.name )
            print( "      Device type:", device_type_as_string( device.type ) )
            print( "      Device Max Buf:", device.image_max_buffer_size )
            print( "      Device Max CUs:", device.max_compute_units )
            print( "      Device Max WIs:", device.max_work_item_sizes )
            print( "      Device Max WGs:", device.max_work_group_size )
            print( "      ------------------------------------------" )

            #ctx  = cl.Context( [ device ] )
            #fila = cl.CommandQueue( ctx )

            # PREPARING TO RUN StructSizes
            #host = np.zeros( 11, np.uint32 )

            #_program    = ctx.get_compiled_kernel("vgl_lib/get_struct_sizes.cl", "get_struct_sizes")
            #kernel_run  = _program.get_struct_sizes
    """
    print( "###",  get_device( cl.device_type.CPU ) )
    print( "###", get_device( cl.device_type.GPU ) )
