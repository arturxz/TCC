/** Maximum or union between two images.

    Maximum or union between img_input1 and img_input2. Result saved in img_output.
  */

#include "vglConst.h"

__kernel void vglClNdBinMax(__global VGL_PACK_CL_SHADER_TYPE* img_input1,
                            __global VGL_PACK_CL_SHADER_TYPE* img_input2,
                            __global VGL_PACK_CL_SHADER_TYPE* img_output)
{
#if __OPENCL_VERSION__ < 200
  int coord = (  (get_global_id(2) - get_global_offset(2)) * get_global_size(1) * get_global_size(0)) +
              (  (get_global_id(1) - get_global_offset(1)) * get_global_size (0)  ) +
                 (get_global_id(0) - get_global_offset(0));
#else
  int coord = get_global_linear_id();
#endif

  VGL_PACK_CL_SHADER_TYPE p1 =  img_input1[coord];
  VGL_PACK_CL_SHADER_TYPE p2 =  img_input2[coord];
  VGL_PACK_CL_SHADER_TYPE result = p1 | p2;
  img_output[coord] = result;
}
