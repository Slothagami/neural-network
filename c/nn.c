#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "nn.h"

Network* make_fc_network(unsigned int *sizes, LayerFunc activation, GradFunc activation_grad, LossFunc loss) {
    //eg: ((28*28, 100, 50, 10), Tanh)
    Network* net = malloc(sizeof(Network));

    net -> loss = loss;
    net -> num_layers = sizeof(sizes) / sizeof(unsigned int) - 1; // (-1) exclude the input, as its not a layer
    net -> layers = malloc(sizeof(Layer) * net -> num_layers * 2); // times two because of activation layers

    for(unsigned int i = 1; i <= net -> num_layers; i++) {
        net -> layers[2 * (i - 1)] = make_layer(sizes[i-1], sizes[i], fc_layer, fc_layer_back);
        net -> layers[2 * (i - 1) + 1] = make_activation_layer(activation, activation_grad);
    }
    return net;
}

void free_network(Network* net) {
    for(unsigned int i = 0; i <= net -> num_layers * 2; i++) {
        free_layer(net -> layers[i]);
    }
    free(net -> layers);
    free(net);
}

mat* layer_forward(Layer* layer, mat* x) {
    return layer ->forward(x, layer -> weights, layer -> biases);
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
    return madd(mdot(x, weights), bias);
}

mat* fc_layer_back(mat* x, mat* weights, mat* bias, mat* out_error, double lr) {
    mat *in_error = mdot(out_error, mtranspose(weights));
    mat *weights_error = mdot(mtranspose(x), out_error);

    weights -> data = msub(weights, mscale(lr, weights_error)) -> data;
    bias    -> data = msub(bias,    mscale(lr, out_error)) -> data;

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
