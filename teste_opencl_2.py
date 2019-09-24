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
    ctx = cl.Context( [ get_device( cl.device_type.GPU ) ] )
    fila = cl.CommandQueue( ctx )

    vl.vglClInit(ctx, None, None)
    if( vl.vglClImage.struct_sizes is not None ):
        print( "### Teste executado com sucesso! ###" )
