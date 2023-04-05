#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "nn.h"

int main() {
	mat *weights = rand_matrix(1, 2);
	mat *bias = rand_matrix(1, 1);
	printm(weights);

	mat *input = new_matrix(2, 1);
	mat *out = fc_layer(input, weights, bias);
	printm(out);

	mfree(weights);
	mfree(bias);
	mfree(input);
	mfree(out);

	return 0;
}
