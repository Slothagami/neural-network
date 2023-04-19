typedef struct {
	double* data;
    unsigned int width;
    unsigned int height;
    unsigned int size;
} mat;

// utility
double rand_norm(double mean, double stdev);

// Constructor like
mat* new_matrix(unsigned int, unsigned int);
mat* rand_norm_matrix(unsigned int width, unsigned int height, double mean, double stdev);
mat* rand_matrix(unsigned int, unsigned int);
mat* matrix_like(mat*);
mat* mcopy(mat*);
void mfree(mat*);

// Operations
mat* mdot(mat*, mat*);

double mmax(mat* matrix);
double margmax(mat* matrix);
double msum(mat* matrix);

mat* mvtile(mat* matrix, unsigned int height);
mat* midentity(unsigned int size);

mat* mscale(double, mat*);
mat* mscalediv(mat*, double divideby);
mat* mscaleadd(double, mat*);
mat* mscalesub(mat*, double);

mat* madd(mat*, mat*);
mat* msub(mat*, mat*);
mat* mmult(mat*, mat*);
mat* mdiv(mat*, mat*);
mat* mmap(double (*func)(double), mat*);
mat* mtranspose(mat*);

// Misc
void mfill(mat*, double);
void printm(mat*);
void printmshape(mat* matrix);
int msame_shape(mat*, mat*);
