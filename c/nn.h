typedef mat* (*LayerFunc)(mat* x, mat* weights, mat* bias);

typedef struct {
    LayerFunc forward;
    LayerFunc backward;
    mat* weights;
    mat* biases;
} Layer;

typedef struct {
    Layer *layers;
    int num_layers;
} Network;

// Network make_simple_network(unsigned int *sizes, LayerFunc activation);
Layer* make_layer(unsigned int in_size, unsigned int out_size, LayerFunc forward, LayerFunc backward);

void free_layer(Layer* layer);

mat* fc_layer(mat* x, mat* weights, mat* bias);
mat* fc_layer_back(mat* x, mat* weights, mat* bias, mat* out_error, double lr);

double mse(mat* target, mat* pred);
mat* mse_grad(mat* target, mat* pred);

mat* mat_tanh(mat* x);
mat* mat_tanh_grad(mat* x);
