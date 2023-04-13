#include <stdio.h>
#include <stdlib.h>
#include "include/matrix.h"
#include "include/nn.h"
#include <time.h>

int main() {
	// make network (mnist) - lr=.01, ((28*28, 100, 50, 10), Tanh)
	srand(time(NULL)); // set the random seed to the time

	mat *input  = rand_matrix(2, 1);
	mat *target = new_matrix(1, 1);
	mfill(target, 0);

	// Network test
	unsigned int layers[] = {2, 3, 1};
	int num_layers = sizeof(layers) / sizeof(unsigned int) - 1; // len(layers) - 1
	Network* net = make_fc_network(layers, num_layers, mat_tanh, mat_tanh_grad, mse_grad);

	// forward
	mat* result = net_forward(net, input);
	printm(result);

	// backward
	double error = mse(target, result);
	printf("Error: %f\n", error);
	net_backward(net, input, result, target, mse_grad, .01);

	free_network(net);
	mfree(result);
	mfree(input);
	mfree(target);

	return 0;
}
