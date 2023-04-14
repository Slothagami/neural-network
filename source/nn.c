#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "../include/matrix.h"
#include "../include/nn.h"

// Network //
void net_train(Network* net, DispErrorFunc errorFunc, mat** batch, mat** labels, int samples, int epochs, double lr, int interval) {
    mat* result;
    double error_sum;
    for(int epoch = 0; epoch < epochs; epoch++) {
        error_sum = 0;
        for(int sample = 0; sample < samples; sample++) {
            result = net_forward(net, batch[sample]);

            // backward
            error_sum += errorFunc(labels[sample], result);
            net_backward(net, batch[sample], result, labels[sample], net -> loss, lr);
        }

        if((epoch + 1) % interval == 0) printf("Epoch %d, Error: %f\n", epoch + 1, error_sum / samples);
    }
	mfree(result);
}

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
        if(net -> layers[i] == NULL) continue;
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
    return layer -> forward(layer, x);
}
mat* layer_back(Layer* layer, mat* out_error, double lr) {
    return layer -> backward(layer, out_error, lr);
}

// Layer Types //
Layer* make_layer(unsigned int in_size, unsigned int out_size, LayerFunc forward, GradFunc backward) {
    Layer* layer = malloc(sizeof(Layer));

    layer -> forward  = forward;
    layer -> backward = backward;

    layer -> weights = rand_matrix(out_size, in_size);
    layer -> biases  = rand_matrix(1, out_size);

    layer -> input  = NULL;
    layer -> output = NULL;

    return layer;
}
Layer* make_activation_layer(LayerFunc forward, GradFunc backward) {
    Layer* layer = malloc(sizeof(Layer));

    layer -> forward  = forward;
    layer -> backward = backward;

    layer -> weights = NULL;
    layer -> biases  = NULL;
    layer -> input   = NULL;
    layer -> output  = NULL;

    return layer;
}
void free_layer(Layer* layer) {
    if(layer -> weights != NULL) mfree(layer -> weights);
    if(layer -> biases  != NULL) mfree(layer -> biases );
    if(layer -> input   != NULL) mfree(layer -> input  );
    free(layer);
}

mat* fc_layer(Layer* layer, mat* x) {
    mat* dot    = mdot(x, layer -> weights);
    mat* bias_T = mtranspose(layer -> biases);
    mat* result = madd(dot, bias_T);

    mfree(dot);
    mfree(bias_T);
    return result;
}
mat* fc_layer_back(Layer* layer, mat* out_error, double lr) {
    mat* weights_T     = mtranspose(layer -> weights);
    mat* x_T           = mtranspose(layer -> input);
    mat *in_error      = mdot(out_error, weights_T);
    mat *weights_error = mdot(x_T,       out_error);

    mat* scaled_w = mscale(lr, weights_error);
    mat* scaled_e = mscale(lr,     out_error);

    mat* scaled_e_T = mtranspose(scaled_e);
    mat* new_w    = msub(layer -> weights,  scaled_w);
    mat* new_b    = msub(layer -> biases, scaled_e_T);

    mfree(scaled_e_T);

    // freeing the weights data on its own results in the new data being freed when calling mfree(new_w/b) so 
    // swapping them so the right ones get freed
    double* weights_dat = layer -> weights -> data;
    double* bias_dat    = layer -> biases  -> data;

    layer -> weights -> data = new_w -> data;
    layer -> biases  -> data = new_b -> data;

    new_w -> data = weights_dat;
    new_b -> data = bias_dat;

    mfree(weights_error);
    mfree(new_w); // also frees weights -> data since its the same pointer
    mfree(new_b); // also frees bias    -> data since its the same pointer
    mfree(scaled_w);
    mfree(scaled_e);
    mfree(x_T);
    mfree(weights_T);

    return in_error;
}

mat* softmax_layer(Layer* layer, mat* x) {
    double max = mmax(x);
    mat* shift = mscalesub(x, max); // shift values down to avoid overflow
    mat* exp_x = mmap(exp, shift);
    mat* result = mscalediv(exp_x, msum(exp_x));

    mfree(shift);
    mfree(exp_x);
    layer -> output = mcopy(result);
    return result;
}
mat* softmax_layer_back(Layer* layer, mat* out_error, double lr) {
    mat* tmp = mvtile(layer -> output, layer -> output -> size);
    mat* out_error_T = mtranspose(out_error);
    mat* tmp_T       = mtranspose(tmp);

    mat* identity = midentity(layer -> output -> size);
    mat* difference = msub(identity, tmp_T);
    mat* scaled     = mmult(tmp, difference);

    mat* result = mdot(scaled, out_error_T);

    mfree(tmp);
    mfree(out_error_T);
    mfree(tmp_T);
    mfree(identity);
    mfree(difference);
    mfree(scaled);
    return result;
}

// Error Functions //
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

// Activations //
mat* mat_tanh(Layer* layer, mat* x) {
    return mmap(tanh, x);
}
mat* mat_tanh_grad(Layer* layer, mat* out_error, double lr) {
    // 1 - tanh(x)^2
    mat* tanh_result       = mat_tanh(NULL, layer -> input);
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

double max(double a, double b) {
    return (a > b)? a : b;
}
double relu(double x) {
    return max(0, x);
}
double relu_grad(double x) {
    return (x > 0)? 1: 0;
}
mat* mat_relu(Layer* layer, mat* x) {
    return mmap(relu, x);
}
mat* mat_relu_grad(Layer* layer, mat* out_error, double lr) {
    mat* d_relu = mmap(relu_grad, layer -> input);
    mat* result = mmult(d_relu, out_error);

    mfree(d_relu);
    return result;
}

double sigmoid(double x) {
    return 1/(1 + exp(-x));
}
double sigmoid_grad(double x) {
    double sig = sigmoid(x);
    return sig * (1 - sig);
}
mat* mat_sigmoid(Layer* layer, mat* x) {
    return mmap(sigmoid, x);
}
mat* mat_sigmoid_grad(Layer* layer, mat* out_error, double lr) {
    mat* d_sig  = mmap(sigmoid_grad, layer -> input);
    mat* result = mmult(d_sig, out_error);

    mfree(d_sig);
    return result;
}
