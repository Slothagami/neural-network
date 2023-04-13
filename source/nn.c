#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "../include/matrix.h"
#include "../include/nn.h"

Network* make_fc_network(unsigned int *sizes, int num_layers, LayerFunc activation, GradFunc activation_grad, LossFunc loss) {
    Network* net = malloc(sizeof(Network));
    if(net == NULL) return NULL;

    net -> loss = loss;
    net -> num_layers = 2 * num_layers; // include activation layes
    net -> layers = malloc(sizeof(Layer*) * net -> num_layers); // times two because of activation layers
    if(net -> layers == NULL) { // if allocation fails
        free(net);
        return NULL;
    }

    // alternating FC and activation layers
    for(int i = 1; i < net -> num_layers - 1; i += 1) {
        net -> layers[2*i - 2] = make_layer(sizes[i-1], sizes[i], fc_layer, fc_layer_back);
        net -> layers[2*i - 1] = make_activation_layer(activation, activation_grad);
    }
    return net;
}
void free_network(Network* net) {
    if(net == NULL) return;
    for(int i = 0; i < net -> num_layers; i++) {
        free_layer(net -> layers[i]);
    }
    free(net -> layers);
    free(net);
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
void net_backward(Network* net, mat* x, mat* output, mat* target, LossFunc loss, double lr) {
    // backprop for each layer
    mat* error = loss(target, output); // initial error
    for(int i = net -> num_layers - 1; i >= 0; i--) {
        // propogate the error
        mat* new_error = layer_back(net -> layers[i], error, lr);
        mfree(error);
        error = new_error;
    }
    mfree(error);
}

mat* layer_forward(Layer* layer, mat* x) {
    layer -> input = mcopy(x);
    return layer -> forward(x, layer -> weights, layer -> biases);
}
mat* layer_back(Layer* layer, mat* out_error, double lr) {
    return layer -> backward(layer -> input, layer -> weights, layer -> biases, out_error, lr);
}

Layer* make_layer(unsigned int in_size, unsigned int out_size, LayerFunc forward, GradFunc backward) {
    Layer* layer = malloc(sizeof(Layer));

    layer -> forward  = forward;
    layer -> backward = backward;

    layer -> weights = rand_matrix(out_size, in_size);
    layer -> biases  = rand_matrix(1, out_size);

    layer -> input = NULL;

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
    mfree(layer -> input  );
    free(layer);
}

mat* fc_layer(mat* x, mat* weights, mat* bias) {
    mat* dot    = mdot(x, weights);
    mat* bias_T = mtranspose(bias);
    mat* result = madd(dot, bias_T);

    mfree(dot);
    mfree(bias_T);
    return result;
}
mat* fc_layer_back(mat* x, mat* weights, mat* bias, mat* out_error, double lr) {
    mat* weights_T     = mtranspose(weights);
    mat* x_T           = mtranspose(x);
    mat *in_error      = mdot(out_error, weights_T);
    mat *weights_error = mdot(x_T,       out_error);

    mat* scaled_w = mscale(lr, weights_error);
    mat* scaled_e = mscale(lr,     out_error);
    mat* new_w    = msub(weights, scaled_w);
    mat* new_b    = msub(bias,    scaled_e);

    free(weights -> data);
    free(bias    -> data);

    weights -> data = new_w -> data;
    bias    -> data = new_b -> data;

    mfree(weights_error);
    mfree(new_w);
    mfree(new_b);
    mfree(scaled_w);
    mfree(scaled_e);
    mfree(x_T);
    mfree(weights_T);

    return in_error;
}

// Error Functions
mat* mse_grad(mat* target, mat* pred) {
    // 2 * (prediction - target) / target.size
    mat* difference        = msub(pred, target);
    mat* difference_double = mscale(2, difference);
    mat* result            = mscalediv(difference_double, target -> size);

    mfree(difference);
    mfree(difference_double);
    return result;
}
double mse(mat* target, mat* pred) {
    // mean((target - pred)^2)
    mat* diff = msub(pred, target);
    mat* square = mmult(diff, diff);

    double sum = 0;
    for(unsigned int i = 0; i < square->size; i++) {
        sum += square->data[i];
    }

    unsigned int size = square -> size; // can't use the size after its freed
    mfree(diff);
    mfree(square);

    return sum / size;
}

mat* mat_tanh(mat* x, mat* weights, mat* bias) {
    return mmap(tanh, x);
}
mat* mat_tanh_grad(mat* x, mat* weights, mat* bias, mat* out_error, double lr) {
    // 1 - tanh(x)^2
    mat* tanh_result       = mat_tanh(x, NULL, NULL);
    mat* tanh_squared      = mmult(tanh_result, tanh_result);
    mat* negative          = mscale(-1, tanh_squared);
    mat* negative_plus_one = mscaleadd(1, negative);
    mat* result            = mmult(negative_plus_one, out_error);

    mfree(tanh_result);
    mfree(tanh_squared);
    mfree(negative);
    mfree(negative_plus_one);
    return result;
}
