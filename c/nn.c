#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "matrix.h"
#include "nn.h"

Network* make_fc_network(unsigned int *sizes, int num_layers, LayerFunc activation, GradFunc activation_grad, LossFunc loss) {
    //eg: ((28*28, 100, 50, 10), Tanh)
    Network* net = malloc(sizeof(Network));

    net -> loss = loss;
    net -> num_layers = 2 * num_layers; // include activation layes
    net -> layers = malloc(sizeof(Layer*) * net -> num_layers); // times two because of activation layers

    // alternating FC and activation layers
    for(int i = 1; i < net -> num_layers - 1; i += 1) {
        net -> layers[2*i - 2] = make_layer(sizes[i-1], sizes[i], fc_layer, fc_layer_back);
        net -> layers[2*i - 1] = make_activation_layer(activation, activation_grad);
    }
    return net;
}

mat* net_forward(Network* net, mat* input) {
    mat* result = mcopy(input);
    for(int i = 0; i < net -> num_layers; i++) {
        mat* new_result = layer_forward(net -> layers[i], result);
        mfree(result);
        result = new_result;
    }
    return result;
}

void free_network(Network* net) {
    for(int i = 0; i < net -> num_layers; i++) {
        free_layer(net -> layers[i]);
    }
    free(net -> layers);
    free(net);
}

mat* layer_forward(Layer* layer, mat* x) {
    return layer -> forward(x, layer -> weights, layer -> biases);
}
void layer_back(Layer* layer, mat* x, mat* target, mat* out_error, double lr) {
    layer -> backward(x, layer -> weights, layer -> biases, out_error, lr);
}

Layer* make_layer(unsigned int in_size, unsigned int out_size, LayerFunc forward, GradFunc backward) {
    Layer* layer = malloc(sizeof(Layer));

    layer -> forward  = forward;
    layer -> backward = backward;

    layer -> weights = rand_matrix(out_size, in_size);
    layer -> biases  = rand_matrix(1, out_size);

    return layer;
}
Layer* make_activation_layer(LayerFunc forward, GradFunc backward) {
    Layer* layer = malloc(sizeof(Layer));

    layer -> forward  = forward;
    layer -> backward = backward;
    layer -> weights = NULL;
    layer -> biases  = NULL;

    return layer;
}

void free_layer(Layer* layer) {
    mfree(layer -> weights);
    mfree(layer -> biases );
    free(layer);
}

mat* fc_layer(mat* x, mat* weights, mat* bias) {
    return madd(mdot(x, weights), mtranspose(bias));
}

mat* fc_layer_back(mat* x, mat* weights, mat* bias, mat* out_error, double lr) {
    mat *in_error = mdot(out_error, mtranspose(weights));
    mat *weights_error = mdot(mtranspose(x), out_error);

    mat* new_w = msub(weights, mscale(lr, weights_error));
    mat* new_b = msub(bias,    mscale(lr,     out_error));

    free(weights -> data);
    free(bias    -> data);
    mfree(weights_error);

    weights -> data = new_w -> data;
    bias    -> data = new_b -> data;

    return in_error;
}

// Error Functions
mat* mse_grad(mat* target, mat* pred) {
    // 2 * (prediction - target) / target.size
    return mscalediv(
        mscale(2, msub(pred, target)),
        target -> size
    );
}

double mse(mat* target, mat* pred) {
    // mean((target - pred)^2)
    mat* diff = msub(pred, target);
    mat* square = mmult(diff, diff);

    double sum = 0;
    for(unsigned int i = 0; i < square->size; i++) {
        sum += square->data[i];
    }

    mfree(diff);
    mfree(square);

    return sum / square->size;
}

mat* mat_tanh(mat* x, mat* weights, mat* bias) {
    return mmap(tanh, x);
}

mat* mat_tanh_grad(mat* x, mat* weights, mat* bias, mat* out_error, double lr) {
    // 1 - tanh(x)^2
    mat* tanh_result = mat_tanh(x, NULL, NULL);
    return mscaleadd(1, mscale(-1, mmult(tanh_result, tanh_result)));
}
