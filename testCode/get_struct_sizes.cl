typedef struct VglClStrEl{ 
    float data[VGL_ARR_CLSTREL_SIZE];
    int ndim;
    int shape[VGL_ARR_SHAPE_SIZE];
    int offset[VGL_ARR_SHAPE_SIZE];
    int size;
} VglClStrEl;

typedef struct VglClShape{ 
    int ndim;
    int shape[VGL_ARR_SHAPE_SIZE];
    int offset[VGL_ARR_SHAPE_SIZE];
    int size;
} VglClShape;

__kernel void get_struct_sizes( __global uint *struct_sizes )
{
    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    VglClStrEl strel;
    VglClShape shape;
    uint base;

    if (global_id == 0){
        base = (uint) &strel;
        // DATA STARTS IN POSITION __
        struct_sizes[0] = (uint) sizeof(strel);
		struct_sizes[1] = (uint) (&strel.data)-base;
		struct_sizes[2] = (uint) (&strel.shape)-base;
		struct_sizes[3] = (uint) (&strel.offset)-base;
		struct_sizes[4] = (uint) (&strel.ndim)-base;
		struct_sizes[5] = (uint) (&strel.size)-base;
		
        base = (uint) &shape;
		struct_sizes[6] = (uint) sizeof(shape);
		struct_sizes[7] = (uint) (&shape.ndim)-base;
		struct_sizes[8] = (uint) (&shape.shape)-base;
		struct_sizes[9] = (uint) (&shape.offset)-base;
		struct_sizes[10] = (uint) (&shape.size)-base;
    }
    return;
}
