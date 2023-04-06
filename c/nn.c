#include <stdlib.h>
#include <math.h>
#include "matrix.h"

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

mat* mat_tanh(mat* x) {
    return mmap(tanh, x);
}

mat* mat_tanh_grad(mat* x) {
    // 1 - tanh(x)^2
    mat* tanh_result = mat_tanh(x);
    return mscaleadd(1, mscale(-1, mmult(tanh_result, tanh_result)));
}
