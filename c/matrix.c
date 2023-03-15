#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

mat* new_matrix(unsigned int width, unsigned int height) {
	mat *matrix = malloc(sizeof(mat));
    
    matrix -> width  = width;
    matrix -> height = height;
    matrix -> size   = width * height;
    matrix -> data   = malloc(sizeof(double) * width * height);
    
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
    
    return prod; // is there a better way than O(n^3)?
}

mat* mscale(double scale, mat* matrix) {
    for(unsigned int i = 0; i < matrix -> size; i++) {
    	matrix -> data[i] *= scale;
    }
    return matrix;
}

mat* madd(mat* a, mat* b) {
    mat *sum = new_matrix(a -> width, a -> height);
    for(unsigned int i = 0; i < a -> size; i++) {
    	sum -> data[i] = b -> data[i] + a -> data[i];
    }
    return sum;
}

mat* mmult(mat* a, mat* b) {
    mat *prod = new_matrix(a -> width, a -> height);
    for(unsigned int i = 0; i < a -> size; i++) {
    	prod -> data[i] = b -> data[i] * a -> data[i];
    }
    return prod;
}

mat* mmap(double (*func)(double), mat* matrix) {
    mat *map = new_matrix(matrix -> width, matrix -> height);
    for(unsigned int i = 0; i < matrix -> size; i++) {
    	map -> data[i] = func(matrix -> data[i]);
    }
    return map;
}
