#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

//#if defined(ACML)
//#include <acml.h>
//#else
#include <mkl_lapack.h>
//#endif  // ACML                                                                                                  

void a_init(double **, int);
int eigenvalues(double**, double *, int , double*, double **);
double randn(double, double);
int hessian_lj(double **, double *, int);
void mc_probability(double *, double *, double,double);
double energy_lj(double, double, double);
double cost_function(double*, double*, int);
double pot_force(double, double*);
  
double** mat_double(int rowdim, int coldim)
{
  double** matrix = NULL;

  /* matrix: array of array of pointers */
  /* matrix: pointer to rows */
  matrix = (double**)malloc(rowdim * sizeof(double*));

  /* matrix[0]: pointer to elements */
  matrix[0] = (double*)malloc(rowdim * coldim * sizeof(double));

  for (int i = 1; i < rowdim; i++)
    matrix[i] = matrix[i - 1] + coldim;

  return matrix;
}

void main(int argc, char *argv[])
{
  ///////////
  // If argv[1] = 1 or 2 (=flag)
  //1 - Run first part(generate pot params), 2 - Run second part (calculate weight from cost)
  int flag; 
  int i;
  int m;
  int params = 2;
  double delta[2] = {0,0};
  double w[params];
  double **a = mat_double(params, params);
  double **z = mat_double(params, params);
  double **z0 = mat_double(params, params);
  double **array;
  
  double sigma = 2.27357076;
  double epsilon = 0.54789329;
  double r = 3.0;
  double parameters[3] = {sigma, epsilon, r};
  int test;

  double eigenvalue = 0.0;
  double lambda[params];
  double R = 0.1; // Value fixed for rescaling to improve acceptance ratio  

  double initial_energy;
  double weight;
  double initial_cost;

  FILE *myFile;

  //read file into array
  char number[256];
  int num_r;
  double *input = (double*)malloc(1 * sizeof(double));
  double cost;
  double temp;
  /////////////////////////

  //  printf( "argc = %d\n", argc );
  // for(i = 0; i < argc; ++i ) {
  // printf( "argv[ %d ] = %s\n", i, argv[i] );
  // }

  if(argc==2){
    flag = atol(argv[1]);
    // printf("FLAG = %d\n",flag);
    if (flag == 1){


  ///////////////////

  /*
  myFile = fopen("reference_forces_041016.txt", "r");
  if(myFile == NULL) {
    perror("Error opening file");

    EXIT_FAILURE;
  }

  i=0;
  while(fgets(number, 256, myFile)){
    forces[i] = atof(number);
    //    printf("%g\n",forces[i]);
    i++;
  }

  fclose(myFile);
  */
  for (int j=0;j<500;j++){
    
    for (i=0;i<params;i++){ w[i]=0; }

    test = hessian_lj(a,parameters,params);

    if (j==0) {
      // initial_energy = energy_lj(parameters[0],parameters[1],parameters[2]);
      //  temp = cost_function(forces,parameters,num_r);
      //temp = 2*temp/2;
      // printf("INITIAL COST = %g\n", temp);
      m = eigenvalues(a,delta,params,w,z0);
    }else{
      m = eigenvalues(a,delta,params,w,z);
    }
    
    // FOR RANDOM NUMBERS BETWEEN RUNS:
    srand(time(NULL));
    
    if(m==params){
      
      for (i=0;i<params;i++)
	{
	  eigenvalue = fabs(w[i]);
	  //     printf("eigenvalue = %g\n",eigenvalue);
	  
	  if (eigenvalue<1){ lambda[i] = 1/sqrt(eigenvalue); }
	  else { lambda[i] = 1; }
	  
	  double r = randn(0.0,1.0);
	  // printf("r = %g\n", r);
	  
	  lambda[i] = lambda[i] * R * r;
	}
      
      
      for (i=0;i<params;i++){
	delta[i] = 0;
	for(int j=0;j<params;j++){
	  
	  delta[i] += z0[i][j]*lambda[j];
	}  
      }
      
      // cost = cost_function(forces,parameters,num_r);
      // printf("%d ",j);
      //mc_probability(parameters,delta, cost,temp);
      
      parameters[0] += delta[0];
      parameters[1] += delta[1];
      
       printf("%g %g\n",parameters[0],parameters[1]);
      
      
    }}    
  
    }
    else if (flag == 2){

      myFile = fopen("cost_column.txt", "r");
      if(myFile == NULL) {
	perror("Error opening file");
	exit;
      }
      
      i=0;
      while(fgets(number, 1024, myFile)){
	
	if (i==0){
	  input[i] = atof(number);
	  i++;
	}
	else{
	  
	  input = realloc(input, (i+1) * sizeof(double));
	  input[i] = atof(number);
	  i++;
	}
      }
      
      fclose(myFile);
      
      num_r = i; //i = number of data points 
      
      
      //  array = mat_double(num_r,3);

      initial_cost = input[2];
      //      printf("BEST FIT INFO: %g %g %g %g\n", input[0], input[1],initial_cost,exp(-1));
      
      for (i=5;i<num_r;i+=3){
	sigma = input[i-2];
	epsilon = input[i-1];
	cost = input[i];
	weight = cost/initial_cost;
	//printf("before: %g\n",weight);
	weight = exp(-weight);
	//printf("after: %g\n",weight);
	
		printf("%g %g %g %g\n",sigma,epsilon,cost,weight);
      }
      
      
      //      free(array);
      
    }
    else{ printf("YOU SHOULDN'T BE HERE - flag == ?\n");}
    
  }else{ printf("YOU SHOULDN'T BE HERE - argc != 2\n");}
  ///////////////////           
  
  free(a);
  free(z);
  free(z0);

  
}

double cost_function(double* forces, double* params, int num_r){

  double cost = 0;
  double force_from_pot;
  double dft_force;
  double dft_r;

   for (int i=0;i<num_r;i++){
    dft_r=forces[i*2];
    dft_force=forces[(2*i)+1];
    //    printf("%g\n",force_from_pot);
    force_from_pot = pot_force(dft_r,params);
    cost += pow(dft_force-force_from_pot,2);

  }
   cost *= 0.5;

   // printf("\nCost = %g\n\n",cost);
  return cost;
}

double pot_force(double r, double* params){

  double force;
  double sigma6 = pow(params[0],6);
  double sigma12 = pow(sigma6,2);
  double r7 = pow(r,7);
  double r13 = pow(r,13);

  
  force = 24*params[1]*((sigma6/r7)-(2*(sigma12/r13)));

  return force;

}

void mc_probability(double* params, double * delta, double cost, double temp){
  
  double probability;
  // double energy_before;
  // double energy_after;
  int    decision; // decision = 1 (accept), = 0 (reject)

  // Set T=1 for now (see Brown & Sethna + MH-MC)
  // P(data|model(params+delta))~exp(-C(params+delta)/T)
  // Simplify for now: P_acc(before->after) = exp(-(E_before - E_after)/k_b*T) (set k_b = T = 1)

  // energy_before = energy_lj(params[0],params[1],params[2]);
  // energy_after = energy_lj(params[0]+delta[0],params[1]+delta[1],params[2]);
  
  // probability = energy_after - energy_before;
  // probability = exp(probability);



  probability = exp(-(cost/temp));
    printf("Cost = %g ",cost);
  printf("probability = %g\n",probability);
  
  //if((probability>1)&&(energy_after>(10*initial_energy))){
  //delta[0] = -delta[0];
  //delta[1] = -delta[1];
  // }

  
 }

double energy_lj(double sigma, double epsilon, double r){

  double energy;
  
  // Hardcode LJ for toy model
  energy = 4*epsilon*( pow((sigma/r),12) - pow((sigma/r),6) );

  return energy;
}

int hessian_lj(double ** const a, double* const params, int number_params){

  double sigma_11 = pow(params[0],11);
  double sigma_5  = pow(params[0],5);
  double sigma_10 = pow(sigma_5,2);
  double sigma_4  = pow(params[0],4);

  double r = params[2];
  double r6 = pow(r,6);
  double r_6 = 1/r6;
  double r_12 = 1/(pow(r6,2));

  a[0][0] = 0;
  a[0][1] = 48*(sigma_11*r_12)-24*(sigma_5*r_6);
  a[1][0] = a[0][1];
  a[1][1] = (params[1]*528*sigma_10*r_12)-(params[1]*120*sigma_4*r_6);

  return number_params;
}

int eigenvalues(double** const a, double *delta, int params, double *w, double** const z){

  /* IN */
char jobz = 'V'; /* Compute eigenvectors and eigenvalues */
char range = 'V'; /* all eigenvalues in the half-open interval (VL,VU] will be found */
char uplo = 'U'; /* Upper triangle of A is stored */
int lda = params; /* leading dimension of the array A. lda >= max(1,N) */
double vl = -3; /* eigenvalue lower bound */ 
double vu = 40; /* eigenvalue upper bound */
double abstol = 0.00001; /* 2*DLAMCH('S');  absolute error tolerance for eigenvalues */
int ldz = params; /* Dimension of array z */
int il = 0;
int iu = 0;

/* OUT */
int m; /* number eigenvalues found */
int iwork[5*params];
int lwork = 8*params;
double work[lwork];
int ifail[params]; /* contains indices of unconverged eigenvectors if info > 0  */
int info = 0; 
int i;

for(i=0;i<params;i++)
{
  ifail[i] = 0;
  iwork[i] = 0;
}

for(i=params;i<5*params;i++)
{
  iwork[i] = 0;
}

 dsyevx_(&jobz, &range, &uplo, &params, &a[0][0], &lda, &vl, &vu, &il, &iu, &abstol, &m, w, &z[0][0], &ldz, work, &lwork, iwork, ifail, &info); 

 return m;
 
}

void a_init(double ** a, int params)
{

  double test[4][4]={{1,2,3,4},{2,6,7,8},{3,7,11,12},{4,8,12,16}};
 
  for (int i = 0;i<params;i++){
    for(int j = 0;j<params;j++)
      {
	a[j][i]=test[i][j];

      }
  }

}


double randn (double mu, double sigma)
{
  /* Using Marsaglia polar method to generate Gaussian distributed random numbers */

  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;

  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }

    do
      {
	U1 = -1 + ((double) rand () / RAND_MAX) * 2;
	U2 = -1 + ((double) rand () / RAND_MAX) * 2;
	W = pow (U1, 2) + pow (U2, 2);
      }
    while (W >= 1 || W == 0);

    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (mu + sigma * (double) X1);
}
