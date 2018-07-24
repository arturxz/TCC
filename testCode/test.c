#include <stdio.h>
#include <stdlib.h>

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

int main(){
	VglClStrEl strel;
    VglClShape shape;
    unsigned int a, i;
    float b;
    unsigned int offset = (unsigned int)sizeof(strel);

    printf("VglStrEl Structure\n");
    i = (unsigned int) &strel;
    printf("data: %u\n", (unsigned int) (&strel.data)-i);
    printf("shape: %u\n", (unsigned int) (&strel.shape)-i);
    printf("offset: %u\n", (unsigned int) (&strel.offset)-i);
    printf("ndim: %u\n", (unsigned int) (&strel.ndim)-i);
    printf("size: %u\n", (unsigned int) (&strel.size)-i);
	
	return 0;
}
