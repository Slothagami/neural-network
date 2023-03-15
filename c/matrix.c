#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

mat* new_matrix(unsigned int width, unsigned int height, double* data) {
	mat *matrix = malloc(sizeof(mat));
    
    matrix -> width  = width;
    matrix -> height = height;
    matrix -> data   = data;
    
    return matrix;
}

void printm(mat* matrix) {
	for(unsigned int y = 0; y < matrix -> height; y++){
		for(unsigned int x = 0; x < matrix -> width; x++){
        	// change print precision, global var?
    		printf("%.1f ", matrix -> data[matrix -> width * y + x]);
    	}
        printf("\n");
    }
}

mat* mdot(mat* a, mat* b) {
	// calculate output size, and allocate memory for the data
	unsigned int width  = b -> width;
    unsigned int height = a -> height;
    double *data = malloc(sizeof(double) * width * height);
    
    // create product matrix
    mat* prod = new_matrix(width, height, data);
    
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
        	data[width * y + x] = sum;
        }
    }
    
    return prod; // is there a better way than O(n^3)?
}

mat* mscale(double scale, mat* matrix) {
	// multiply each element by a scalar
	unsigned int size = matrix -> width * matrix -> height;
    for(unsigned int i = 0; i < size; i++) {
    	matrix -> data[i] *= scale;
    }
    return matrix;
}

mat* madd(mat* a, mat* b) {
	unsigned int size  = a -> width * a -> height;

    double *data = malloc(sizeof(double) * size);
    mat *sum = new_matrix(a -> width, a -> height, data);

    for(unsigned int i = 0; i < size; i++) {
    	data[i] = b -> data[i] + a -> data[i];
    }
    return sum;
}

mat* mmult(mat* a, mat* b) {
    unsigned int size  = a -> width * a -> height;

    double *data = malloc(sizeof(double) * size);
    mat *prod = new_matrix(a -> width, a -> height, data);

    for(unsigned int i = 0; i < size; i++) {
    	data[i] = b -> data[i] * a -> data[i];
    }
    return prod;
}

mat* mmap(double (*func)(double), mat* matrix) {
    unsigned int size  = matrix -> width * matrix -> height;

    double *data = malloc(sizeof(double) * size);
    mat *map = new_matrix(matrix -> width, matrix -> height, data);

    for(unsigned int i = 0; i < size; i++) {
    	data[i] = func(matrix -> data[i]);
    }
    return map;
}
