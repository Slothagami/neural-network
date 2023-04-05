#include <stdlib.h>
#include "matrix.h"

mat* fc_layer(mat* x, mat* weights, mat* bias) {
    return madd(mdot(x, weights), bias);
}

mat* fc_layer_back(mat* x, mat* weights, mat* bias, mat* out_error, double lr) {
    mat *in_error = mdot(out_error, mtranspose(weights));
    mat *weights_error = mdot(mtranspose(x), out_error);

    weights = madd(weights, mscale(lr, weights_error));
    bias    = madd(bias,    mscale(lr, out_error));

    return in_error;
}

mat* mse_grad(mat* target, mat* pred) {
    // 2 * (prediction - target) / target.size
    return mscalediv(
        mscale(2, msub(pred, target)),
        target -> size
    );
}
