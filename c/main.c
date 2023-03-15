#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

double func(double);

int main() {
	double a_data[] = {1, 2, 3, 4};
	mat *a = new_matrix(2, 2);
    a -> data = a_data;
    
    double b_data[] = {6, 5, 4, 7};
    mat *b = new_matrix(2, 2);
    b -> data = b_data;
    
    mat* prod = mmap(func, a);

	printm(prod);
    
	mfree(a);
    mfree(b);
    mfree(prod);

	return 0;
}

double func(double x) {
    return x - 1;
}
