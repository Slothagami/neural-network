#include <stdio.h>
#include <stdlib.h>
#include "../include/matrix.h"
#include "../include/nn.h"
#include <time.h>

int main() {
	// make network (mnist) - lr=.01, ((28*28, 100, 50, 10), Tanh)
	srand(time(NULL)); // set the random seed to the time

	// input data (xor inputs)
	int samples = 4;
	mat* a = new_matrix(2, 1);
	mat* b = new_matrix(2, 1);
	mat* c = new_matrix(2, 1);
	mat* d = new_matrix(2, 1);
	mfill(a, 0);
	mfill(b, 0);
	mfill(c, 0);
	mfill(d, 1);

	b -> data[0] = 1;
	c -> data[1] = 1;

	mat* batch[] = {a, b, c, d};

	// labels
	mat* one  = new_matrix(1, 1);
	mat* zero = new_matrix(1, 1);
	mfill(one,  1);
	mfill(zero, 0);
	mat* labels[] = {zero, one, one, zero};

	// train network
	unsigned int layers[] = {2, 3, 1};
	int num_layers = sizeof(layers) / sizeof(unsigned int) - 1; // len(layers) - 1
	Network* net = make_fc_network(layers, num_layers, mat_tanh, mat_tanh_grad, mse_grad);

	net_train(net, mse, mse_grad, batch, labels, samples, 500, .1, 50);

	// test network
	for(int i = 0; i < samples; i++) {
		mat* pred = net_forward(net, batch[i]);
		printm(labels[i]);
		printm(pred);
		printf("\n");
		mfree(pred);
	}

	mfree(a);
	mfree(b);
	mfree(c);
	mfree(d);
	mfree(one);
	mfree(zero);

	free_network(net);

	return 0;
}
