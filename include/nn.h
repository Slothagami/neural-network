typedef mat* (*LayerFunc)(mat* x, mat* weights, mat* bias);
typedef mat* (*GradFunc)(mat* x, mat* weights, mat* bias, mat* out_error, double lr);
typedef mat* (*LossFunc)(mat* target, mat* pred);
typedef double (*DispErrorFunc)(mat* target, mat* pred);

typedef struct {
    LayerFunc forward;
    GradFunc backward;
    mat* weights;
    mat* biases;
    mat* input;
} Layer;

typedef struct {
    Layer **layers;
    LossFunc loss;
    int num_layers;
} Network;

// Network //
void net_train(Network* net, DispErrorFunc errorFunc, mat** batch, mat** labels, int samples, int epochs, double lr, int interval);
Network* make_fc_network(unsigned int *sizes, int num_layers, LayerFunc activation, GradFunc activation_grad, LossFunc loss);
mat* net_forward(Network* net, mat* x);
void net_backward(Network* net, mat* x, mat* output, mat* target, LossFunc loss, double lr);

// Layer //
Layer* make_layer(unsigned int in_size, unsigned int out_size, LayerFunc forward, GradFunc backward);
Layer* make_activation_layer(LayerFunc forward, GradFunc backward);
mat* layer_forward(Layer* layer, mat* x);
mat* layer_back(Layer* layer, mat* out_error, double lr);

void free_layer(Layer*);
void free_network(Network*);

mat* fc_layer(mat* x, mat* weights, mat* bias);
mat* fc_layer_back(mat* x, mat* weights, mat* bias, mat* out_error, double lr);

mat* softmax_layer(mat* x, mat* weights, mat* bias);
mat* softmax_layer_back(mat* x, mat* weights, mat* bias, mat* out_error, double lr);

// Error Functions //
double mse(mat* target, mat* pred);
mat* mse_grad(mat* target, mat* pred);

// Activations //
mat* mat_tanh(mat* x, mat* weights, mat* bias);
mat* mat_tanh_grad(mat* x, mat* weights, mat* bias, mat* out_error, double lr);

double max(double a, double b);
double relu(double x);
double relu_grad(double x);
mat* mat_relu(mat* x, mat* weights, mat* bias);
mat* mat_relu_grad(mat* x, mat* weights, mat* bias, mat* out_error, double lr);

double sigmoid(double x);
double sigmoid_grad(double x);
mat* mat_sigmoid(mat* x, mat* weights, mat* bias);
mat* mat_sigmoid_grad(mat* x, mat* weights, mat* bias, mat* out_error, double lr);
