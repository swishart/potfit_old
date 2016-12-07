int uncertainty_quantification(double);

double randn(double, double);

double** mat_double_mem(int, int);

double** calc_hessian(double);

double calc_pot_params(double** const, double** const, double*, double, double*, int*);

int mc_moves(double**,double*, double*, int, double);

int calc_h0_eigenvectors(double**, double, double, double**, double*);
