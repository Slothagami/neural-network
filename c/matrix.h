typedef struct {
	double* data;
    unsigned int width;
    unsigned int height;
    unsigned int size;
} mat;

// Constructor like
mat* new_matrix(unsigned int, unsigned int);
mat* matrix_like(mat*);
void mfree(mat*);

// Operations
mat* mdot(mat*, mat*);
mat* mscale(double, mat*);
mat* madd(mat*, mat*);
mat* mmult(mat*, mat*);
mat* mmap(double (*func)(double), mat*);

// Misc
void printm(mat*);
