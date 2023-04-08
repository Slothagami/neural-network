#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"

mat* new_matrix(unsigned int width, unsigned int height) {
	mat *matrix = malloc(sizeof(mat));
    
    matrix -> width  = width;
    matrix -> height = height;
    matrix -> size   = width * height;
    matrix -> data   = malloc(sizeof(double) * width * height);
    
    return matrix;
}

mat* mcopy(mat* matrix) {
    mat* new = matrix_like(matrix);
    for(unsigned int i = 0; i < matrix -> size; i++) {
        new -> data[i] = matrix -> data[i];
    }
    return new;
}

mat* rand_matrix(unsigned int width, unsigned int height) {
    mat *matrix = new_matrix(width, height);
    for(unsigned int i = 0; i < width * height; i++) {
        matrix -> data[i] = (float) rand() / RAND_MAX;
    }
    return matrix;
}

void mfree(mat* matrix) {
    if(matrix == NULL) return;
    free(matrix -> data);
    free(matrix);
}

mat* matrix_like(mat* template) {
    return new_matrix(template -> width, template -> height);
}

void mfill(mat* matrix, double value) {
    for(unsigned int i = 0; i < matrix -> size; i++) {
        matrix -> data[i] = value;
    }
}

void printm(mat* matrix) {
	for(unsigned int y = 0; y < matrix -> height; y++){
		for(unsigned int x = 0; x < matrix -> width; x++){
        	// change print precision, global var?
    		printf("%.4f ", matrix -> data[matrix -> width * y + x]);
    	}
        printf("\n");
    }
    printf("\n");
}

int msame_shape(mat* a, mat* b) {
    if(a -> width  != b -> width ) return 0;
    if(a -> height != b -> height) return 0;
    return 1;
}

mat* mdot(mat* a, mat* b) {
    printf("%d %d\n", b -> height, a -> width);

    // assert the shapes can be multiplied
    assert(b -> height == a -> width);

	// calculate output size, and allocate memory for the data
	unsigned int width  = b -> width;
    unsigned int height = a -> height;

    // create product matrix
    mat* prod = new_matrix(width, height);
    
    // calculate matrix product
    for(unsigned int x = 0; x < width; x++) {
    	for(unsigned int y = 0; y < height; y++) {
        	// a goes across, b goes down, calculate weighted sum
            double sum = 0;
            for(unsigned int ind = 0; ind < a -> width; ind++) {
            	double a_val = a -> data[a -> width * y + ind];
            	double b_val = b -> data[b -> width * ind + x];
                
                sum += a_val * b_val;
            }
        
        	// can still use data because its a pointer
        	prod -> data[width * y + x] = sum;
        }
    }
    
    return prod;
}

mat* mscale(double scale, mat* matrix) {
    mat *scaled = matrix_like(matrix);
    for(unsigned int i = 0; i < matrix -> size; i++) {
    	scaled -> data[i] = scale * matrix -> data[i];
    }
    return scaled;
}
mat* mscalediv(mat* matrix, double divideby) {
    return mscale(1/divideby, matrix);
}
mat* mscaleadd(double scale, mat* matrix) {
    mat *scaled = matrix_like(matrix);
    for(unsigned int i = 0; i < matrix -> size; i++) {
    	scaled -> data[i] = scale + matrix -> data[i];
    }
    return scaled;
}
mat* mscalesub(mat* matrix, double scale) {
    mat *scaled = matrix_like(matrix);
    for(unsigned int i = 0; i < matrix -> size; i++) {
    	scaled -> data[i] = matrix -> data[i] - scale;
    }
    return scaled;
}

mat* madd(mat* a, mat* b) {
    assert(msame_shape(a, b));
    mat *sum = matrix_like(a);
    for(unsigned int i = 0; i < a -> size; i++) {
    	sum -> data[i] = b -> data[i] + a -> data[i];
    }
    return sum;
}
mat* msub(mat* a, mat* b) {
    return madd(a, mscale(-1, b));
}

mat* mmult(mat* a, mat* b) {
    assert(msame_shape(a, b));
    mat *prod = matrix_like(a);
    for(unsigned int i = 0; i < a -> size; i++) {
    	prod -> data[i] = b -> data[i] * a -> data[i];
    }
    return prod;
}
mat* mdiv(mat* a, mat* b) {
    assert(msame_shape(a, b));
    mat *prod = matrix_like(a);
    for(unsigned int i = 0; i < a -> size; i++) {
    	prod -> data[i] = b -> data[i] / a -> data[i];
    }
    return prod;
}

mat* mmap(double (*func)(double), mat* matrix) {
    mat *map = matrix_like(matrix);
    for(unsigned int i = 0; i < matrix -> size; i++) {
    	map -> data[i] = func(matrix -> data[i]);
    }
    return map;
}

mat* mtranspose(mat* matrix) {
    mat* transpose = new_matrix(matrix -> height, matrix -> width);

    // rearrange the data, calculate transposed index
    for(unsigned int i = 0; i < matrix -> size; i++) {
        unsigned int x = i % matrix -> width;
        unsigned int y = i / matrix -> width;

        // flip the x and y
        transpose -> data[x * transpose -> width + y] = matrix -> data[i];
    }
    return transpose;
}
