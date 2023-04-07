#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "nn.h"

int main() {
	// make network (mnist) - lr=.01, ((28*28, 100, 50, 10), Tanh)
	mat *weights = rand_matrix(1, 2);
	mat *bias    = rand_matrix(1, 1);
	mat *input   = rand_matrix(2, 1);
	mat *target = new_matrix(1, 1);
	mfill(target, 1);

	mat* out;
	mat* error;

	for(int i = 0; i < 15; i++) {
		out = fc_layer(input, weights, bias);
		printf("error: %f\n", mse(target, out));

		error = mse_grad(target, out);
		fc_layer_back(input, weights, bias, error, .1);
	}

	mfree(weights);
	mfree(bias);
	mfree(input);
	mfree(out);
	mfree(target);
	mfree(error);

	return 0;
}
