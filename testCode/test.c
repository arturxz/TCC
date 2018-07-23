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
    int a;
    float b;
    unsigned int offset = (unsigned int)sizeof(strel);
	
	printf("\t data %u\n", (unsigned int) &(strel.data));
	printf("\t ndim %u\n", (unsigned int) &(strel.ndim)-offset);
	printf("\t shape %u\n", (unsigned int) &(strel.shape)-offset);
	printf("\t offset %u\n", (unsigned int) &(strel.offset)-offset);
	printf("\t size %u\n", (unsigned int) &(strel.size)-offset);
	
	printf("Shape: %u\n", (unsigned int) sizeof(shape));
	
	return 0;
}
