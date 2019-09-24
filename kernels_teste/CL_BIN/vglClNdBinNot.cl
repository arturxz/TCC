/** Negation of binary image img_input. Result is stored in img_output.

  */

#include "vglConst.h"

__kernel void vglClNdBinNot(__global VGL_PACK_CL_SHADER_TYPE* img_input,
                            __global VGL_PACK_CL_SHADER_TYPE* img_output)
{
#if __OPENCL_VERSION__ < 200
  int coord = (  (get_global_id(2) - get_global_offset(2)) * get_global_size(1) * get_global_size(0)) +
              (  (get_global_id(1) - get_global_offset(1)) * get_global_size (0)  ) +
                 (get_global_id(0) - get_global_offset(0));
#else
  int coord = get_global_linear_id();
#endif

  VGL_PACK_CL_SHADER_TYPE p = img_input[coord];
  img_output[coord] = VGL_PACK_MAX_UINT & ~p;
}
