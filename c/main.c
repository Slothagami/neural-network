#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

double func(double);

int main() {
	mat *a = rand_matrix(2, 3);
	printm(a);
	mfree(a);

	return 0;
}

double func(double x) {
    return x - 1;
}
