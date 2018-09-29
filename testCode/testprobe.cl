#include "vglClShape.h"
#include "vglClStrEl.h"

__kernel void testprobe(__global unsigned char* img_input, 
                        __global unsigned char* img_output,  
                        __constant VglClShape* img_shape,
                        __constant VglClStrEl* window)
{
#if __OPENCL_VERSION__ < 200
  int coord = (  (get_global_id(2) - get_global_offset(2)) * get_global_size(1) * get_global_size(0)) +
              (  (get_global_id(1) - get_global_offset(1)) * get_global_size (0)  ) +
                 (get_global_id(0) - get_global_offset(0));
#else
  int coord = get_global_linear_id();
#endif

  
  int ires;
  int idim;
  ires = coord;
  float result = 0.0;
  int img_coord[VGL_ARR_SHAPE_SIZE];
  int win_coord[VGL_ARR_SHAPE_SIZE];
  unsigned int i = 0;
  if (coord == 0){
	printf("########## IN KERNEL ##########\n");
	printf("StrEl Data:\n");
	for(i=0; i< 256; i++){
		printf("%.2f ", window->data[i]);
		if(i > 0 && (i % 24) == 0 ){
			printf("\n");
		}
	}
	printf("\n");
	
	printf("-> StrEl ndim: %i\n", window->ndim);
	
	printf("StrEl Shape:\n");
	unsigned int i = 0;
	for(i=0; i< 20; i++){
		printf("%i ", window->shape[i]);
		if(i > 0 && (i % 10) == 0 ){
			printf("\n");
		}
	}
	printf("\n");
	printf("StrEl Offset:\n");
	for(i=0; i< 20; i++){
		printf("%i ", window->offset[i]);
		if(i > 0 && (i % 10) == 0 ){
			printf("\n");
		}
	}
	printf("\n");
	printf("-> StrEl size: %i\n", window->size);
  }
  img_output[coord] = (unsigned char)125;
}
