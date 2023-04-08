#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "nn.h"
#include <time.h>

int main() {
	// make network (mnist) - lr=.01, ((28*28, 100, 50, 10), Tanh)
	srand(time(NULL)); // set the random seed to the time

	mat *input  = rand_matrix(2, 1);
	mat *target = new_matrix(1, 1);
	mfill(target, 1);

	// Network struct test
	unsigned int layers[] = {2, 5, 1};
	int num_layers = sizeof(layers) / sizeof(unsigned int) - 1; // len(layers) - 1
	Network* net = make_fc_network(layers, num_layers, mat_tanh, mat_tanh_grad, mse_grad);

	// forward pass
	mat* result = net_forward(net, input);
	printm(result);

	free_network(net);
	mfree(result);


	// Single Layer Test
	Layer* layer = make_layer(2, 1, fc_layer, fc_layer_back);	

	mat* out;
	mat* error;

	for(int i = 0; i < 5; i++) {
		out = layer_forward(layer, input);
		printf("error: %f\n", mse(target, out));

		error = mse_grad(target, out);
		layer_back(layer, input, target, error, .1);
	}

	free_layer(layer);
	mfree(input);
	mfree(out);
	mfree(target);
	mfree(error);

	return 0;
}
