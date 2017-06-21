int uncertainty_quantification(double, const char*);

double** calc_hessian(double, int);

double calc_pot_params(double** const, double** const, double*, double, double*, int*, FILE*);

int mc_moves(double**,double*, double*, int, double, FILE*);

int calc_h0_eigenvectors(double**, double, double, double**, double*, int);

double** mat_double(int, int); //in powell_lsq.c

int calc_svd(double**, double**, double*, int);