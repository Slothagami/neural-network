typedef struct {
	double* data;
    unsigned int width;
    unsigned int height;
    unsigned int size;
} mat;

void printm(mat*);
mat* new_matrix(unsigned int, unsigned int);
mat* matrix_like(mat*);
// do i need a matrix free method too?
mat* mdot(mat*, mat*);
mat* mscale(double, mat*);
mat* madd(mat*, mat*);
mat* mmult(mat*, mat*);
mat* mmap(double (*func)(double), mat*);
