#include <stdlib.h>
#include "matrix.h"

mat* fc_layer(mat* x, mat* weights, mat* bias) {
    return madd(mdot(x, weights), bias);
}

mat* fc_layer_back(mat* x, mat* weights, mat* bias, mat* label) {
    
}
