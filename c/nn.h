mat* fc_layer(mat* x, mat* weights, mat* bias);
mat* fc_layer_back(mat* x, mat* weights, mat* bias, mat* out_error, double lr);

double mse(mat* target, mat* pred);
mat* mse_grad(mat* target, mat* pred);
