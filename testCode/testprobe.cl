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
	// SHAPE DATA
	printf("-> Shape ndim: %d\n", img_shape->ndim);	
	printf("Shape Shape:\n");
	for(i=0; i< VGL_ARR_SHAPE_SIZE; i++){
		printf("%d ", img_shape->shape[i]);
	}
	printf("\n");
	printf("Shape Offset:\n");
	for(i=0; i< VGL_ARR_SHAPE_SIZE; i++){
		printf("%d ", img_shape->offset[i]);
	}
	printf("\n");
	printf("-> Shape size: %d\n", img_shape->size);	
	
	// STREL DATA
	printf("StrEl Data:\n");
	for(i=0; i< VGL_ARR_CLSTREL_SIZE; i++){
		printf("%.2f ", window->data[i]);
		if(i > 0 && (i % 24) == 0 ){
			printf("\n");
		}
	}
	printf("\n");
	printf("-> StrEl ndim: %i\n", window->ndim);
	printf("StrEl Shape:\n");
	unsigned int i = 0;
	for(i=0; i< VGL_ARR_SHAPE_SIZE; i++){
		printf("%d ", window->shape[i]);
	}
	printf("\n");
	printf("StrEl Offset:\n");
	for(i=0; i< VGL_ARR_SHAPE_SIZE; i++){
		printf("%d ", window->offset[i]);
	}
	printf("\n");
	printf("-> StrEl size: %d\n", window->size);
  }
  img_output[coord] = (unsigned char)125;
}
