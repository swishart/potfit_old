#include <mkl_lapack.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "uq.h"

#include "potfit.h"

#include "force.h"

#include "random.h"

#if defined(UQ)&&(APOT) //Only for analytic potentials at the moment


double** mat_double_mem(int rowdim, int coldim)
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

double** calc_hessian(double cost_0){

  //If smooth cutoff is enabled, there is an extra parameter (h), which we are not adjusting
  int num_params = g_pot.opt_pot.idxlen;
    if (g_pot.smooth_pot[0] == 1) {num_params -= 1;}

  // Create the Hessian of analytic potential parameters
  // For N parameters, require:
  // diagonal: 2N cost evaluations
  // off-diagonal: 2N(N-1) cost evaluations (4 per hessian element)

  // Allocate memory to store:
  // - the cost evaluations per hessian element
  // - the size of each parameter perturbation (i.e. 0.0001*parameter)
  // - the final hessian elements
  double param_perturb_dist[num_params]; 
  double** hessian = mat_double_mem(num_params, num_params); //mat_double() defined in powell_lsq.c
  double two_cost0 = 2*cost_0;
  
  for (int j=0;j<num_params;j++){
    param_perturb_dist[j] = 0.0001*g_pot.opt_pot.table[g_pot.opt_pot.idx[j]];
  }
  
  // For diagonal entries, use (c_(i+1) - 2*cost_0 + c_(i-1))/(param_perturb_dist[i]^2)
  for (int i=0;i<num_params;i++){
    
    double cost_plus;
    double cost_minus;

    g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] += param_perturb_dist[i];
    cost_plus = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

    g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] -= 2*param_perturb_dist[i];
    cost_minus = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

    g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] += param_perturb_dist[i];
       
    hessian[i][i] = cost_plus - two_cost0 + cost_minus;
    hessian[i][i] /= (param_perturb_dist[i]*param_perturb_dist[i]);
  }

  // For off-diagonal entries:
  // Use [c_(i+1)(j+1)-c_(i+1)(j-1)-c_(i-1)(j+1)+c_(i-1)(j-1)]/(param_perturb_dist[i]*param_perturb_dist[j]*4)
  for (int i=0;i<num_params;i++){
    for (int j=(i+1);j<num_params;j++){

      double cost_2plus;
      double cost_2minus;
      double cost_pm;
      double cost_mp;

      // c_(i+1)(j+1)
      g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] += param_perturb_dist[i];
      g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] += param_perturb_dist[j];
      cost_2plus = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

      // c_(i+1)(j-1)
      g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] -= 2*param_perturb_dist[j];
      cost_pm = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);
      
      // c_(i-1)(j+1)
      g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] -= 2*param_perturb_dist[i];
      g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] += 2*param_perturb_dist[j];
      cost_mp = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

      // c_(i-1)(j-1)
      g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] -= 2*param_perturb_dist[j];
      cost_2minus = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);
           
      hessian[i][j] = cost_2plus + cost_2minus - cost_pm - cost_mp;
      hessian[i][j] /= (4*param_perturb_dist[i]*param_perturb_dist[j]);

      hessian[j][i] = hessian[i][j];
    }
  }
  return hessian;
}


int calc_h0_eigenvectors(double** h_0, double vl, double vu, double** v_0, double* w){

  //If smooth cutoff is enabled, there is an extra parameter (h), which we are not adjusting
  int params = g_pot.opt_pot.idxlen;
    if (g_pot.smooth_pot[0] == 1) {params -= 1;}

  char jobz = 'V'; /* Compute eigenvectors and eigenvalues */
  char range = 'V'; /* all eigenvalues in the half-open interval (VL,VU] will be found */
  char uplo = 'U'; /* Upper triangle of A is stored */
  int lda = params; /* leading dimension of the array A. lda >= max(1,N) */
  double abstol = 0.00001; /* 2*DLAMCH('S');  absolute error tolerance for eigenvalues */
  int ldz = params; /* Dimension of array z */
  int il = 0;
  int iu = 0;

  int m; /* number eigenvalues found */
  int iwork[5*params];
  int lwork = 8*params;
  double work[lwork];
  int ifail[params]; /* contains indices of unconverged eigenvectors if info > 0  */
  int info = 0;
  int i;
  //  double w[params];
 
  dsyevx_(&jobz, &range, &uplo, &params, &h_0[0][0], &lda, &vl, &vu, &il, &iu, &abstol, &m, w, &v_0[0][0], &ldz, work, &lwork, iwork, ifail,&info);
  
  return m;
}




double calc_pot_params(double** const a, double** const v_0, double* cost_before, double cost_0, double* w, int* weight){

  //If smooth cutoff is enabled, there is an extra parameter (h), which we are not adjusting
  int params = g_pot.opt_pot.idxlen;
    if (g_pot.smooth_pot[0] == 1) {params -= 1;}
  
  char jobz = 'V'; /* Compute eigenvectors and eigenvalues */
  char range = 'V'; /* all eigenvalues in the half-open interval (VL,VU] will be found */
  char uplo = 'U'; /* Upper triangle of A is stored */
  int lda = params; /* leading dimension of the array A. lda >= max(1,N) */
  double vl = -1; /* eigenvalue lower bound */
  double vu = 1; /* eigenvalue upper bound */
  double abstol = 0.00001; /* 2*DLAMCH('S');  absolute error tolerance for eigenvalues */
  int ldz = params; /* Dimension of array z */
  int il = 0;
  int iu = 0;

  int m; /* number eigenvalues found */
  int iwork[5*params];
  int lwork = 8*params;
  double work[lwork];
  int ifail[params]; /* contains indices of unconverged eigenvectors if info > 0  */
  int info = 0;
  int i;
  
  double **z = mat_double_mem(params, params);
  //double w[params];

  //  dsyevx_(&jobz, &range, &uplo, &params, &a[0][0], &lda, &vl, &vu, &il, &iu, &abstol, &m, w, &z[0][0], &ldz, work, &lwork, iwork, ifail,&info);

  // store old parameters incase proposed move isn't accepted
  double old_params[params];
  for (int i=0;i<params;i++){
    old_params[i] = g_pot.opt_pot.table[g_pot.opt_pot.idx[i]];
  }

  int count = 1;
  int mc_decision = mc_moves(v_0, w, cost_before, params, cost_0);

  // Keep generating trials for this hessian until a move is accepted
  // This saves multiple calculations of the same hessian when a move isn't accepted
  while (mc_decision == 0) {

   count++; //If move not accepted, count current parameters again
    
    //reset parameters to initials params
    for (int i=0;i<params;i++){
      g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] = old_params[i];
    }
    //    printf("--- %g %g %g\n",g_pot.opt_pot.table[g_pot.opt_pot.idx[0]], g_pot.opt_pot.table[g_pot.opt_pot.idx[1]], *cost_before);

    //call function recursively until we accept a move for this set of eigenvalues
    mc_decision = mc_moves(v_0, w, cost_before, params, cost_0);

  }
  *weight = count;

  free(z);
  // Return new cost
  return *cost_before;
}


int mc_moves(double** v_0,double* w, double* cost_before, int m, double cost_0) {

  //If smooth cutoff is enabled, there is an extra parameter (h), which we are not adjusting
  int params = g_pot.opt_pot.idxlen;
    if (g_pot.smooth_pot[0] == 1) {params -= 1;}
  
  double lambda[params];
  double R = sqrt(0.5); // FIX THIS FOR NOW
  double cost_after;
  
  // If not all eigenvalues are found (i.e. m != params), replace them with 1.
  // THIS SHOULD NOT HAPPEN NOW
  for (int i = m; i<params; i++)
    {
      double r = R * normdist();
      lambda[i] = r;
    }

  // If eigenvalue is less than 1, replace it with 1.
  for (int i=0;i<m;i++){
    if (w[i] < 1.0){
      // printf("replacing small eigenvalue %g with 1. \n",w[i]);
      w[i] = 1.0;
    }
  }
  
  
  for (int i=0;i<m;i++)
    {
      double r = R * normdist();
      w[i] = fabs(w[i]);
      lambda[i] = 1/sqrt(w[i]);
      lambda[i] *= r;
      printf("%g %g ", r,lambda[i]);
    }

  // Matrix multiplication (delta_param[i] = Sum{1}{params} [v_0[i][j] * (r[j]/lambda[j])] )
  for (int i=0;i<params;i++){
    for(int j=0;j<params;j++){
      g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] += v_0[i][j]*lambda[j];
    }
  }
  
  cost_after = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

  double cost_diff = cost_after - *cost_before;

  // Accept downhill moves outright
  if (cost_diff < 0){
    *cost_before = cost_after;
    return 1;
  }
  
  // Monte Carlo step (seeded from srand(time(NULL)) in calc_pot_params() )
  // Acceptance probability = 0.8
  // generate uniform random number [0,1], if greater than 0.8 then accept change
  // if step accepted, move new cost to cost_before for next cycle
  double probability = exp(-(params*(cost_diff))/(2*cost_0));
  double mc_rand_number = eqdist();

  if (mc_rand_number <= probability){
    *cost_before = cost_after;
    return 1;
  }


  // Print out unseccessful moves
  //  printf("%g %g %g 1 1\n",g_pot.opt_pot.table[g_pot.opt_pot.idx[0]], g_pot.opt_pot.table[g_pot.opt_pot.idx[1]], cost_after);

  /********************/
  for(int i=0;i<params;i++){
    printf("%g ",g_pot.opt_pot.table[g_pot.opt_pot.idx[i]]);
  }
  printf("%g 1 1\n", cost_after);
  /*******************/
  
  // If move not accepted, return 0. 
  return 0;
}

#endif  // UQ&&APOT
