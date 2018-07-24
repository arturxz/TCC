typedef struct VglClStrEl{ 
    float data[256];
    int ndim;
    int shape[20];
    int offset[20];
    int size;
} VglClStrEl;

typedef struct VglClShape{ 
    int ndim;
    int shape[20];
    int offset[20];
    int size;
} VglClShape;

__kernel void get_struct_sizes( __global uint *struct_sizes )
{
    const uint global_id = get_global_id(0u)+get_global_id(1u)*get_global_size(0u);
    VglClStrEl strel;
    VglClShape shape;
    uint base;

    printf("In GPU (probing):\n Kernel instance = %d\n", global_id);
	
	/* VERSAO ANTERIOR JA MEXIDA
    if (global_id==0) {
        offset = (uint)&(vgclstrel.data);
        struct_sizes[0] = (uint)sizeof(vgclstrel);
        struct_sizes[1] = (uint)&(vgclstrel.data)-offset;
        struct_sizes[2] = (uint)&(vgclstrel.ndim)-offset;
        struct_sizes[3] = (uint)&(vgclstrel.shape)-offset;
        struct_sizes[4] = (uint)&(vgclstrel.offset)-offset;
        struct_sizes[5] = (uint)&(vgclstrel.size)-offset;
        offset = (uint)&(vgclshape.ndim);
        struct_sizes[6] = (uint)sizeof(vgclshape);
        struct_sizes[7] = (uint)&(vgclshape.ndim)-offset;
        struct_sizes[8] = (uint)&(vgclshape.shape)-offset;
        struct_sizes[9] = (uint)&(vgclshape.offset)-offset;
        struct_sizes[10] =(uint)&(vgclshape.size)-offset;
    }
    */
    
    if (global_id == 0){
        base = (unsigned int) &strel;
		struct_sizes[0] = (uint) (&strel.data)-base;
		struct_sizes[1] = (uint) (&strel.shape)-base;
		struct_sizes[2] = (uint) (&strel.offset)-base;
		struct_sizes[3] = (uint) (&strel.ndim)-base;
		struct_sizes[4] = (uint) (&strel.size)-base;
		
        base = (unsigned int) &shape;
		struct_sizes[5] = (uint) (&shape.ndim)-base;
		struct_sizes[6] = (uint) (&shape.shape)-base;
		struct_sizes[7] = (uint) (&shape.offset)-base;
		struct_sizes[8] = (uint) (&shape.size)-base;
		
		printf("struct_sizes[0]: %u\n", struct_sizes[0]);
		printf("struct_sizes[1]: %u\n", struct_sizes[1]);
		printf("struct_sizes[2]: %u\n", struct_sizes[2]);
		printf("struct_sizes[3]: %u\n", struct_sizes[3]);
		printf("struct_sizes[4]: %u\n", struct_sizes[4]);
		printf("struct_sizes[5]: %u\n", struct_sizes[5]);
		printf("struct_sizes[6]: %u\n", struct_sizes[6]);
		printf("struct_sizes[7]: %u\n", struct_sizes[7]);
		printf("struct_sizes[8]: %u\n", struct_sizes[8]);
    }
    return;
}
