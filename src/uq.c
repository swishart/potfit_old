/****************************************************************                                                                     
 *                                                                                                                                     
 * uq.c: Uncertainty quantification using sloppy model method                                                                                                                                                                                                                
 ****************************************************************                                                                      
 *                                                                                                                                     
 * Copyright 2002-2017 - the potfit development team                                                                                   
 *                                                                                                                                     
 * http://potfit.sourceforge.net/                                                                                                      
 *                                                                                                                                     
 ****************************************************************                                                                      
 *                                                                                                                                     
 * This file is part of potfit.                                                                                                        
 *                                                                                                                                     
 * potfit is free software; you can redistribute it and/or modify                                                                      
 * it under the terms of the GNU General Public License as published by                                                                
 * the Free Software Foundation; either version 2 of the License, or                                                                   
 * (at your option) any later version.                                                                                                 
 *                                                                                                                                     
 * potfit is distributed in the hope that it will be useful,                                                                           
 * but WITHOUT ANY WARRANTY; without even the implied warranty of                                                                      
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                                                       
 * GNU General Public License for more details.                                                                                        
 *                                                                                                                                     
 * You should have received a copy of the GNU General Public License                                                                   
 * along with potfit; if not, see <http://www.gnu.org/licenses/>.                                                                      
 *                                                                                                                                     
 ****************************************************************/

#include "potfit.h"

#if defined(MKL)
#include <mkl_lapack.h>
#elif defined(ACML)
#include <acml.h>
#elif defined(__ACCELERATE__)
#include <Accelerate/Accelerate.h>
#else
#error No math library defined!
#endif  // ACML

#include "uq.h"
#include "force.h"
#include "random.h"

#define VERY_SMALL 1.E-12

#if defined(UQ)&&(APOT) /* Only for analytic potentials at the moment */

/****************************************************************
 *
 *   main sloppy model routine
 *
 ****************************************************************/
void ensemble_generation(double cost_0) {
  
  /* open file */
  FILE* outfile = fopen(g_files.sloppyfile, "w");
  if (outfile == NULL)
    error(1, "Could not open file %s for writing\n", filename);

  /* Initialise variables to 0 */
  int pot_attempts      = 0;
  double acc_prob       = 0.00;
  
  /* Calculate the best fit hessian */
  double** hessian = calc_hessian(cost_0, g_pot.opt_pot.idxlen);
  
  int m     = 0;
  double vl = -1; // Initial lower bound for eigenvalues - this range is adjusted until all are found
  double vu = 1;  // Initial upper bound for eigenvalues
  int count = 0;

  /* Allocate memory for the eigenvectors */
  double** v_0 = mat_double(g_pot.opt_pot.idxlen,g_pot.opt_pot.idxlen);
  double eigenvalues[g_pot.opt_pot.idxlen];

  /* Enter loop to find eigenvalues, increasing the range by a factor of ten each until all eigenvalues are found. 
     If after 5 iterations the eigenvalues have not be found, revert to signular value decomposition */
  while (m < g_pot.opt_pot.idxlen) {

    vl *= 10;
    vu *= 10;

    m = calc_h0_eigenvectors(hessian, vl, vu, v_0, eigenvalues, g_pot.opt_pot.idxlen);
    
    if (count > 5){
      fprintf(outfile,"NOT CONVERGING! Use singular value decomposition \n");

      m = calc_svd(hessian, v_0, eigenvalues, g_pot.opt_pot.idxlen);

      /* Check all eigenvalues have been found, otherwise exit */
      if (m != g_pot.opt_pot.idxlen){
        error(1, "Only %d of %d eigenvalues were found!", m, g_pot.opt_pot.idxlen);
      }
    }
    count +=1;
  }

  for (int i=0;i<g_pot.opt_pot.idxlen;i++){
    if (eigenvalues[i] < 0){
      warning("Negative eigenvalue of %.4f, has the best fit minimum been found?!\nThis implies the best fit potential is not at the miminum!\n", eigenvalue[i]);
    }
  }

  /* Print eigenvalues and eigenvectors of hessian to sloppyfile */
  fprintf(outfile,"------------------------------------------------------\n");
  fprintf(outfile,"Eigenvalues and eigenvectors of the best fit hessian:\n");
  fprintf(outfile, "eigenvalue, eigenvector (x, y ,z)\n");
  for (int i=0;i<g_pot.opt_pot.idxlen;i++){
    fprintf(outfile,"%.4f\t",eigenvalues[i]);
    for (int j=0;j<g_pot.opt_pot.idxlen;j++){
      fprintf(outfile,"%.4f ",v_0[i][j]);
    }
    fprintf(outfile,"\n");
  }

  /* Print initial best fit */
  for (int i=0;i<g_pot.opt_pot.idxlen;i++){
  fprintf(outfile, "Parameter %d\t",i+1);
  }
  fprintf(outfile, "Cost\tWeight\tAccepted\tAttempts\tAcceptance Probability\n", );
  for (int i=0;i<g_pot.opt_pot.idxlen;i++){
    fprintf(outfile,"%g\t",g_pot.opt_pot.table[g_pot.opt_pot.idx[i]]);
  }
  fprintf(outfile,"%g\t1\t1\t%d\t%.2f\n", cost_0, pot_attempts, acc_prob);

  /* Initialise variables and take first Monte Carlo step */
  int weight      = 1;
  int* weight_ptr = &weight;

  /* run until number of moves specified in param file are accepted */
  for (int i=0; i<=g_param.acc_moves;i++)
    {
      double cost = generate_mc_sample(hessian, v_0, cost, cost_0, eigenvalues, weight_ptr, outfile);

      pot_attempts += weight;
      acc_prob = (((double)i+1.0))/(double)pot_attempts; /* Add one to include the MC step outside loop */

      /* Write accepted move to file */
      for(int i=0;i<g_pot.opt_pot.idxlen;i++){
      fprintf(outfile,"%g\t",g_pot.opt_pot.table[g_pot.opt_pot.idx[i]]);
      }
      fprintf(outfile,"%g\t%d\t1\t%d\t%.2f\n", cost, weight, pot_attempts, acc_prob);
      
    }

fclose(outfile);
printf("UQ ensemble parameters written to %s\n", filename);
}

/****************************************************************
 *
 *    Calculate the best fit potential hessian
 *
 ****************************************************************/

double** calc_hessian(double cost){
/*  Implementing equation 5.7.10 from Numerical recipes in C
  
  Create the Hessian of analytic potential parameters
  For N parameters, require:
  diagonal: 2N cost evaluations
  off-diagonal: 2N(N-1) cost evaluations (4 per hessian element)

  Allocate memory to store:
  - the cost evaluations per hessian element
  - the size of each parameter perturbation (i.e. 0.0001*parameter)
  - the final hessian elements */
  double param_perturb_dist[g_pot.opt_pot.idxlen; 
  double** hessian    = mat_double(g_pot.opt_pot.idxlen, g_pot.opt_pot.idxlen); /* mat_double() defined in powell_lsq.c */
  double two_cost     = 2*cost;
  double perturbation = 0.0001;  

  /* Pre-calculate each parameter perturbation */
  for (int j=0;j<g_pot.opt_pot.idxlen;j++){
    param_perturb_dist[j] = perturbation * g_pot.opt_pot.table[g_pot.opt_pot.idx[j]];

    if (g_pot.opt_pot.table[_pot.opt_pot.idx[j]] == 0){
      param_perturb_dist[j] = perturbation;
      warning("parameter %d is 0. Using set perturbation of %f.\n", j, perturbation);
    }
  }
  
  /* For diagonal entries, use (c_(i+1) - 2*cost + c_(i-1))/(param_perturb_dist[i]^2) */
  for (int i=0;i<g_pot.opt_pot.idxlen;i++){
    
    double cost_plus;
    double cost_minus;

    g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] += param_perturb_dist[i];
    cost_plus = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

    g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] -= 2*param_perturb_dist[i];
    cost_minus = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

    g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] += param_perturb_dist[i];
       
    /* print a warning if either cost_plus or cost_minus are less than 10^(-12) */
    if (cost_plus < VERY_SMALL) || (cost_minus < VERY_SMALL){
      warning("The change in cost_plus/cost_minus when calculating the hessian is less than 10^(-12). This will affect precision.\nConsider changing the scale of cost perturbation. \n");

    }

    hessian[i][i] = cost_plus - two_cost + cost_minus;
    hessian[i][i] /= (param_perturb_dist[i]*param_perturb_dist[i])
  }

  /* For off-diagonal entries:
     Use [c_(i+1)(j+1)-c_(i+1)(j-1)-c_(i-1)(j+1)+c_(i-1)(j-1)]/(param_perturb_dist[i]*param_perturb_dist[j]*4) */
  for (int i=0;i<g_pot.opt_pot.idxlen;i++){
    for (int j=(i+1);j<g_pot.opt_pot.idxlen;j++){

      double cost_2plus;
      double cost_2minus;
      double cost_pm;
      double cost_mp;

      /* c_(i+1)(j+1) */
      g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] += param_perturb_dist[i];
      g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] += param_perturb_dist[j];
      cost_2plus = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

      /* c_(i+1)(j-1) */
      g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] -= 2*param_perturb_dist[j];
      cost_pm = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);
      
      /* c_(i-1)(j+1) */
      g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] -= 2*param_perturb_dist[i];
      g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] += 2*param_perturb_dist[j];
      cost_mp = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

      /* c_(i-1)(j-1) */
      g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] -= 2*param_perturb_dist[j];
      cost_2minus = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

      /* reset to start */
      g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] += param_perturb_dist[i];
      g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] += param_perturb_dist[j];
      
      hessian[i][j] = cost_2plus + cost_2minus - cost_pm - cost_mp;
      hessian[i][j] /= (4*param_perturb_dist[i]*param_perturb_dist[j]);

      hessian[j][i] = hessian[i][j];
    }
  }
  return hessian;
}

/***********************************************************************
 *
 *    Find hessian eigenvalues and eigenvectors by eigen decomposition
 *
 ***********************************************************************/

int calc_h0_eigenvectors(double** hessian, double lower_bound, double upper_bound, double** v_0, double* w){

  char jobz = 'V'; /* Compute eigenvectors and eigenvalues */
  char range = 'V'; /* all eigenvalues in the half-open interval (VL,VU] will be found */
  char uplo = 'U'; /* Upper triangle of A is stored */
  int lda = g_pot.opt_pot.idxlen; /* leading dimension of the array A. lda >= max(1,N) */
  double abstol = 0.00001; /* 2*DLAMCH('S');  absolute error tolerance for eigenvalues */
  int ldz = g_pot.opt_pot.idxlen; /* Dimension of array z */
  int il = 0;
  int iu = 0;

  int m; /* number eigenvalues found */
  int iwork[5*g_pot.opt_pot.idxlen];
  int lwork = 8*g_pot.opt_pot.idxlen;
  double work[lwork];
  int ifail[g_pot.opt_pot.idxlen]; /* contains indices of unconverged eigenvectors if info > 0  */
  int info = 0;
  int i;
  
  #if defined(MKL)
    dsyevx_(&jobz, &range, &uplo, &g_pot.opt_pot.idxlen, &hessian[0][0], &lda, &lower_bound, &upper_bound, &il, &iu, 
      &abstol, &m, w, &v_0[0][0], &ldz, work, &lwork, iwork, ifail, &info);
  #elif defined(ACML)
    dsyevx_(&jobz, &range, &uplo, &g_pot.opt_pot.idxlen, &hessian[0][0], &lda, &lower_bound, &upper_bound, &il, &iu, 
      &abstol, &m, w, &v_0[0][0], &ldz, work, &lwork, iwork, ifail, &info, int jobz_len, int range_len, int uplo_len);  
  #elif defined(__ACCELERATE__)
        dsyevx_(&jobz, &range, &uplo, &g_pot.opt_pot.idxlen, &hessian[0][0], &lda, &lower_bound, &upper_bound, &il, &iu, 
      &abstol, &m, w, &v_0[0][0], &ldz, work, &lwork, iwork, ifail, &info);
  #endif


  return m;
}

/********************************************************
 *
 *    Find hessian eigenvalues and eigenvectors by SVD
 *
 *********************************************************/

int calc_svd(double** hessian, double** u, double* s){

  char jobu = 'A'; /* Compute left singular vectors */
  char jobvt = 'A'; /* Compute right singular vectors */
  int lda = g_pot.opt_pot.idxlen; /* leading dimension of the array A. lda >= max(1,N) */
  double vl = 0;
  double vu = 0;
  int il = 0;
  int iu = 0;

  int ns; /* number singular values found */
  int iwork[12*g_pot.opt_pot.idxlen];
  int lwork = 5*g_pot.opt_pot.idxlen;
  double work[lwork];
  int info = 0;
  double vt[g_pot.opt_pot.idxlen][g_pot.opt_pot.idxlen];
  

  #if defined(MKL)
      dgesvd_(&jobu, &jobvt, &g_pot.opt_pot.idxlen, &g_pot.opt_pot.idxlen, &hessian[0][0], &lda, s, &u[0][0],
       &lda, &vt[0][0], &g_pot.opt_pot.idxlen, work, &lwork, &info);
  #elif defined(ACML)
      dgesvd_(&jobu, &jobvt, &g_pot.opt_pot.idxlen, &g_pot.opt_pot.idxlen, &hessian[0][0], &lda, s, &u[0][0],
       &lda, &vt[0][0], &g_pot.opt_pot.idxlen, work, &lwork, &info);
  #elif defined(__ACCELERATE__)
      dgesvd_(&jobu, &jobvt, &g_pot.opt_pot.idxlen, &g_pot.opt_pot.idxlen, &hessian[0][0], &lda, s, &u[0][0],
       &lda, &vt[0][0], &g_pot.opt_pot.idxlen, work, &lwork, &info);
  #endif



  if (info == 0){
    return lda;
  }else{
    printf("Finding all eigenvalues by singular value decomposition (SVD) unsuccessful.\n");
  }
  
}

/********************************************************
 *
 *    Initiate MC steps until one is accepted
 *
 *********************************************************/

double generate_mc_sample(double** const a, double** const v_0, double cost_before, double cost_0, double* w, int* weight, FILE* outfile){

  /* If smooth cutoff is enabled, there is an extra parameter (h), which we are adjusting unless it's bounds are chosen to be fixed */

  /* store old parameters incase proposed move isn't accepted */
  double old_params[g_pot.opt_pot.idxlen];

  for (int i=0;i<g_pot.opt_pot.idxlen;i++){
    old_params[i] = g_pot.opt_pot.table[g_pot.opt_pot.idx[i]];
  }

  int count = 1;
  int mc_decision = mc_moves(v_0, w, cost_before, cost_0, outfile);

  /* Keep generating trials for this hessian until a move is accepted
     This saves multiple calculations of the same hessian when a move isn't accepted */
  while (mc_decision == 0) {

   count++; /* If move not accepted, count current parameters again */
    
    /* reset parameters to initials params */
    for (int i=0;i<pg_pot.opt_pot.idxlen;i++){
      g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] = old_params[i];
    }
   
    /* call function recursively until we accept a move for this set of eigenvalues */
    mc_decision = mc_moves(v_0, w, cost_before, cost_0, outfile);

  }
  *weight = count;
  
  /* Return new cost */
  return cost_before;
}

/********************************************************
 *
 *    Take MC step and decide if accepted
 *
 *********************************************************/

int mc_moves(double** v_0,double* w, double cost_before, double cost_0, FILE* outfile) {
  
  double lambda[g_pot.opt_pot.idxlen];
  double R; 
  double cost_after;
  double delta[g_pot.opt_pot.idxlen];

  R = sqrt(g_param.acceptance_rescaling);
  
  /* If eigenvalue is less than 1, replace it with 1. */
  for (int i=0;i<g_pot.opt_pot.idxlen;i++){
#if defined(MIN_STEP)
    if (w[i] > 1.0){ w[i] = 1.0; }
#else /* Use max(lambda,1) */
    if (w[i] < 1.0){ w[i] = 1.0; }
#endif
    double r = R * normdist();
    w[i] = fabs(w[i]);
    lambda[i] = 1/sqrt(w[i]);
    lambda[i] *= r;
  }
  
  /* Matrix multiplication (delta_param[i] = Sum{1}{g_pot.opt_pot.idxlen} [v_0[i][j] * (r[j]/lambda[j])] ) */

  for (int i=0;i<g_pot.opt_pot.idxlen;i++){
    delta[i] = 0;
    for(int j=0;j<g_pot.opt_pot.idxlen;j++){
      delta[i] += v_0[i][j]*lambda[j];
    }
    g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] += delta[i];
  }
  
  cost_after = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

  double cost_diff = cost_after - cost_before;

  /* Accept downhill moves outright */
  if (cost_diff < 0){
    *cost_before = cost_after;

    /* Print change in parameters */
    for(int i=0;i<g_pot.opt_pot.idxlen;i++){
      printf("%.4f ",delta[i]);
    }
    printf("\n");

    return 1;
  }
  
  /* Monte Carlo step (seeded from srand(time(NULL)) in generate_mc_sample() )
     Acceptance probability = 0.8
     generate uniform random number [0,1], if greater than 0.8 then accept change
     if step accepted, move new cost to cost_before for next cycle */
  double probability = exp(-(g_pot.opt_pot.idxlen*(cost_diff))/(2*cost_0));
  double mc_rand_number = eqdist();

  if (mc_rand_number <= probability){
    cost_before = cost_after;

    /* Print change in parameters */
    for(int i=0;i<g_pot.opt_pot.idxlen;i++){
      printf("%.4f ",delta[i]);
    }
    printf("\n");
    
    return 1;
  }

  /* Print out unsuccessful moves */
  for(int i=0;i<g_pot.opt_pot.idxlen;i++){
    fprintf(outfile,"%g ",g_pot.opt_pot.table[g_pot.opt_pot.idx[i]]);
  }
  fprintf(outfile,"%g 1 0 - -\n", cost_after);
  
  /* If move not accepted, return 0 */
  return 0;
}

#endif  /* UQ&&APOT */