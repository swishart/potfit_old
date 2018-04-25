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
#include <sys/stat.h>

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
#include "errors.h"
#include "random.h"
#include "potential_output.h"
#include "potential_input.h"


#define VERY_SMALL 1.E-12
#define VERY_LARGE 1.E12

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
    error(1, "Could not open file %s for writing\n", g_files.sloppyfile);

  /* Initialise variables to 0 */
  int pot_attempts      = 0;
  double acc_prob       = 0.00;
  
  /* Calculate the best fit hessian */
  double** hessian = calc_hessian(cost_0, 1);
  
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

    m = calc_h0_eigenvectors(hessian, vl, vu, v_0, eigenvalues);
    
    if (count > 5){
      printf("NOT CONVERGING! Use singular value decomposition.\n");

      m = calc_svd(hessian, v_0, eigenvalues);

      /* Check all eigenvalues have been found, otherwise exit */
      if (m != g_pot.opt_pot.idxlen){
        error(1, "Only %d of %d eigenvalues were found!", m, g_pot.opt_pot.idxlen);
      }
    }
    count +=1;
  }

  for (int i=0;i<g_pot.opt_pot.idxlen;i++){
    if (eigenvalues[i] < 0){
      warning("Negative eigenvalue of %.4f, has the best fit minimum been found?!\nThis implies the best fit potential is not at the miminum!\n", eigenvalues[i]);
    }
  }

  /* Print eigenvalues and eigenvectors of hessian to sloppyfile */
  fprintf(outfile,"------------------------------------------------------\n\n");
  fprintf(outfile,"Eigenvalues and eigenvectors of the best fit hessian:\n");
  fprintf(outfile,"eigenvalue, eigenvector (x, y ,z)\n");
  for (int i=0;i<g_pot.opt_pot.idxlen;i++){
    fprintf(outfile,"%-11.4f ",eigenvalues[i]);
    for (int j=0;j<g_pot.opt_pot.idxlen;j++){
      fprintf(outfile,"%-10.4f ",v_0[i][j]);
    }
    fprintf(outfile,"\n");
  }

  fprintf(outfile,"\n------------------------------------------------------\n\n");
  /* Print initial best fit */
  fprintf(outfile, "Identifier ");
  for (int i=0;i<g_pot.opt_pot.idxlen;i++){
  fprintf(outfile, "Param %-4d ",i+1);
  }
  fprintf(outfile, "Cost       Weight     Accepted   Attempts   Acceptance Probability\n");

  // /* Create directory to store parameter files */ -SW 
  // int status;
  // status = mkdir(g_files.output_prefix, S_IRWXU | S_IRWXG | S_IRWXO);
  // if (status != 0){
  //   error(1,"Could not create directory '%s' for potential files.\n", dirname);
  // }

  /* Initialise variables and take first Monte Carlo step */
  int weight      = 1;
  int* weight_ptr = &weight;

  /* run until number of moves specified in param file are accepted */
  for (int i=0; i<=g_param.acc_moves;i++)
    {

      if (i != 0) {
        /* Write weight from of previous parameter set - i.e. how many trials before this new parameter set was accepted */
        fprintf(outfile,"%-10d 1          %-10d %-10.2f\n", weight, pot_attempts, acc_prob); 
      }
      
      if (i == g_param.acc_moves) {
        /* For the final configuration (i = (g_param.acc_moves - 1) print the remaining weight calculated and then exit */
        continue;
      }

      double cost = generate_mc_sample(hessian, v_0, cost, cost_0, eigenvalues, weight_ptr, outfile);

      pot_attempts += weight;
      acc_prob = (((double)i+1.0))/(double)pot_attempts; /* Add one to include the MC step outside loop */

      /* Write accepted move to file */
      fprintf(outfile,"%-10d", i+1);
      for(int i=0;i<g_pot.opt_pot.idxlen;i++){
      fprintf(outfile,"%-10.6g ",g_pot.opt_pot.table[g_pot.opt_pot.idx[i]]);
      }
      fprintf(outfile,"%-10.6g ", cost);

      /* Write potential input file for parameter ensemble */
#if !defined(NO_SLOPPY)
      char file[255];
      char end[255];
      strcpy(file, g_files.output_prefix);
      sprintf(end,".sloppy_pot_%d",i+1);
      strcat(file, end);
      write_pot_table_potfit(file); 

      if (cost < cost_0) {
        warning("New best fit parameter set found in %s. Old cost = %5.4g, new cost = %5.4g\n",file, cost_0,cost);
      }
#else 
      if (cost < cost_0) {
        warning("New best fit parameter set found for potential %d. Old cost = %5.4g, new cost = %5.4g\n",i+1, cost_0,cost);
      }
#endif



      
    }

fclose(outfile);
printf("UQ ensemble parameters written to %s\n", g_files.sloppyfile);
}

/****************************************************************
 *
 *    Calculate the best fit potential hessian
 *
 ****************************************************************/

double** calc_hessian(double cost, int counter){
/*  Implementing equation 5.7.10 from Numerical recipes in C
  
  Create the Hessian of analytic potential parameters
  For N parameters, require:
  diagonal: 2N cost evaluations
  off-diagonal: 2N(N-1) cost evaluations (4 per hessian element)

  Allocate memory to store:
  - the cost evaluations per hessian element
  - the size of each parameter perturbation (i.e. 0.0001*parameter)
  - the final hessian elements */
  double param_perturb_dist[g_pot.opt_pot.idxlen]; 
  double** hessian    = mat_double(g_pot.opt_pot.idxlen, g_pot.opt_pot.idxlen); /* mat_double() defined in powell_lsq.c */
  double two_cost     = 2.0 * cost;
  //  double perturbation = 0.0001;  
  double perturbation = 2.0 / g_pot.opt_pot.idxlen;
  int counter_max = 10;
  double new_cost_param_values[g_pot.opt_pot.idxlen+1];
  
  /* Check that we haven't been through this thing 10 times, if so error */
  if( counter == counter_max ){
    error(1, "Too many recalculations of the hessian implies the potential is poorly fit.\n It is advised to rerun parameter optimisation and use the true minimum.\n");
  }

  /* Initialise values for possible better fit found */
  for (int j=0;j<g_pot.opt_pot.idxlen;j++){
    new_cost_param_values[j] = 0;
  }
  new_cost_param_values[g_pot.opt_pot.idxlen+1] = VERY_LARGE;

//double perturbation = floor(cost/g_pot.opt_pot.idxlen);
  printf("perturbation is %f\n", perturbation);
  fflush(stdout);

  /* Pre-calculate each parameter perturbation */
  for (int j=0;j<g_pot.opt_pot.idxlen;j++){
    param_perturb_dist[j] = perturbation * g_pot.opt_pot.table[g_pot.opt_pot.idx[j]];
    
    if (g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] == 0){
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

    if ((cost_plus < cost) && (cost_plus < new_cost_param_values[g_pot.opt_pot.idxlen+1])) {
      //      warning("A new cost minimum has been found.\nOriginal cost = %f,\t New: cost_plus = %f.\nCalculation continuing although it is advised to rerun parameter optimisation and use the true minimum.\n",cost, cost_plus);

      /* If new minima is found, store these values */
      for (int j=0;j<g_pot.opt_pot.idxlen;j++){
	new_cost_param_values[j] = g_pot.opt_pot.table[g_pot.opt_pot.idx[j]];
      }
      new_cost_param_values[g_pot.opt_pot.idxlen+1] = cost_plus;

    }

    g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] -= 2*param_perturb_dist[i];
    cost_minus = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

    if ((cost_minus < cost) && (cost_minus < new_cost_param_values[g_pot.opt_pot.idxlen+1])) {
      //  warning("A new cost minimum has been found.\nOriginal cost = %f,\t New: cost_minus = %f.\nCalculation continuing although it is advised to rerun parameter optimisation and use the true minimum.\n",cost, cost_minus);

      /* If new minima is found, store these values */
      for (int j=0;j<g_pot.opt_pot.idxlen;j++){
        new_cost_param_values[j] = g_pot.opt_pot.table[g_pot.opt_pot.idx[j]];
      }
      new_cost_param_values[g_pot.opt_pot.idxlen+1] = cost_minus;

    }

    /* Reset original param values without perturbation */
    g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] += param_perturb_dist[i];
       
    /* print a warning if either cost_plus or cost_minus are less than 10^(-12) or a new minima is found */
    if ((cost_plus < VERY_SMALL) || (cost_minus < VERY_SMALL)) {
      warning("The change in cost_plus/cost_minus when calculating the hessian is less than 10^(-12).\n This will affect precision. Consider changing the scale of cost perturbation. \n");
    }

    hessian[i][i] = cost_plus - two_cost + cost_minus;
    hessian[i][i] /= (param_perturb_dist[i]*param_perturb_dist[i]);
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

      if ((cost_pm < cost) && (cost_pm < new_cost_param_values[g_pot.opt_pot.idxlen+1])) {
	//warning("A new cost minimum has been found.\nOriginal cost = %f,\t New: cost_pm = %f.\nCalculation continuing although it is advised to rerun parameter optimisation and use the true minimum.\n",cost, cost_pm);

	/* If new minima is found, store these values */
	for (int j=0;j<g_pot.opt_pot.idxlen;j++){
	  new_cost_param_values[j] = g_pot.opt_pot.table[g_pot.opt_pot.idx[j]];
	}
	new_cost_param_values[g_pot.opt_pot.idxlen+1] = cost_pm;
      }

      /* c_(i-1)(j+1) */
      g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] -= 2*param_perturb_dist[i];
      g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] += 2*param_perturb_dist[j];
      cost_mp = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

      if ((cost_mp < cost) && (cost_mp < new_cost_param_values[g_pot.opt_pot.idxlen+1])) {
        //warning("A new cost minimum has been found.\nOriginal cost = %f,\t New: cost_mp = %f.\nCalculation continuing although it is advised to rerun parameter optimisation and use the true minimum.\n",cost, cost_mp);

        /* If new minima is found, store these values */
        for (int j=0;j<g_pot.opt_pot.idxlen;j++){
          new_cost_param_values[j] = g_pot.opt_pot.table[g_pot.opt_pot.idx[j]];
        }
        new_cost_param_values[g_pot.opt_pot.idxlen+1] = cost_mp;

      }

      /* c_(i-1)(j-1) */
      g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] -= 2*param_perturb_dist[j];
      cost_2minus = calc_forces(g_pot.opt_pot.table, g_calc.force, 0);

      /* reset to start */
      g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] += param_perturb_dist[i];
      g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] += param_perturb_dist[j];


      /* print a warning if either cost_pm or cost_mp are less than 10^(-12) or a new minima is found */
      if ((cost_pm < VERY_SMALL) || (cost_mp < VERY_SMALL)) {
	 warning("The change in cost_pm/cost_mp when calculating the hessian is less than 10^(-12).\nThis will affect precision. Consider changing the scale of cost perturbation. \n");
      }
      
      hessian[i][j] = cost_2plus + cost_2minus - cost_pm - cost_mp;
      hessian[i][j] /= (4*param_perturb_dist[i]*param_perturb_dist[j]);

      hessian[j][i] = hessian[i][j];  
    }
  }

  /************* IF A NEW COST VALUE IF FOUND ****************/
  /* If new cost value is found, return parameters */
  if(new_cost_param_values[g_pot.opt_pot.idxlen+1] != VERY_LARGE){
    warning("A new cost minimum has been found.\nOriginal cost = %f,\t New cost = %f.\nCalculation restarting with new best fit potential values.\n\n",cost, new_cost_param_values[g_pot.opt_pot.idxlen+1]);

    printf("NEW COST MINIMA VALUES:\n");
    for(int j=0;j<g_pot.opt_pot.idxlen;j++){
      printf("Param %d = %.8lf\n", j, new_cost_param_values[j]);
    }
    printf("Cost = %f\n", new_cost_param_values[g_pot.opt_pot.idxlen+1]);
    fflush(stdout);


    /* Move old potential to temp file */
    if (g_files.tempfile && strlen(g_files.tempfile)) {
#if defined(APOT)
      update_apot_table(g_pot.opt_pot.table);
#endif  // APOT
      write_pot_table_potfit(g_files.tempfile);
    }

    /* Set new cost potential parameters as the best fit potential  */
    for(int j=0;j<g_pot.opt_pot.idxlen;j++){
      g_pot.opt_pot.table[g_pot.opt_pot.idx[j]] = new_cost_param_values[j];  
    }

    /* Write out new end potential */
#if defined(APOT)
    update_apot_table(g_pot.opt_pot.table);
#endif  // APOT
    write_pot_table_potfit(g_files.endpot);
  
    // will not work with MPI
#if defined(PDIST) && !defined(MPI)
    write_pairdist(&g_pot.opt_pot, g_files.distfile);
#endif  // PDIST && !MPI

    // write the error files for forces, energies, stresses, ...
    write_errors(g_calc.force, new_cost_param_values[g_pot.opt_pot.idxlen+1]);

    /* Rerun hessian calculation with new cost minima */
    hessian = calc_hessian(new_cost_param_values[g_pot.opt_pot.idxlen+1], counter+1);
  }
  /************* IF A NEW COST VALUE IF FOUND - END *************/

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
  double* cost_ptr = &cost_before;

  for (int i=0;i<g_pot.opt_pot.idxlen;i++){
    old_params[i] = g_pot.opt_pot.table[g_pot.opt_pot.idx[i]];
  }

  int count = 0;
  int mc_decision = 0; 

  /* Keep generating trials for this hessian until a move is accepted
     This saves multiple calculations of the same hessian when a move isn't accepted */
  while (mc_decision == 0) {

   count++; /* If move not accepted, count current parameters again */
    
    /* reset parameters to initials params */
    for (int i=0;i<g_pot.opt_pot.idxlen;i++){
      g_pot.opt_pot.table[g_pot.opt_pot.idx[i]] = old_params[i];
    }
   
    /* call function continuously until we accept a move for this set of eigenvalues */
    mc_decision = mc_moves(v_0, w, cost_ptr, cost_0, outfile);

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

int mc_moves(double** v_0,double* w, double* cost_before, double cost_0, FILE* outfile) {
  
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

  double cost_diff = cost_after - *cost_before;

  /* Accept downhill moves outright */
  if (cost_diff < 0){
    *cost_before = cost_after;

#if defined(DEBUG)
    /* Print change in parameters */
    for(int i=0;i<g_pot.opt_pot.idxlen;i++){
      printf("%.4f ",delta[i]);
    }
    printf("\n");
#endif

    return 1;
  }
  
  /* Monte Carlo step (seeded from srand(time(NULL)) in generate_mc_sample() )
     generate uniform random number [0,1], if less than probability then accept change
     if step accepted, move new cost to cost_before for next cycle */
  double probability = exp(-(g_pot.opt_pot.idxlen*(cost_diff))/(2*cost_0));

  double mc_rand_number = eqdist();

  if (mc_rand_number <= probability){
    *cost_before = cost_after;

    printf("prob is %f, random_number is %f, cost_diff is %f\n", probability, mc_rand_number, cost_diff);
    fflush(stdout);

#if defined(DEBUG)
    /* Print change in parameters */
    for(int i=0;i<g_pot.opt_pot.idxlen;i++){
      printf("%.4f ",delta[i]);
    }
    printf("\n");
#endif
    
    return 1;
  }


#if defined(DEBUG)
  /* Print out unsuccessful moves */
  fprintf(outfile,"-          ");
  for(int i=0;i<g_pot.opt_pot.idxlen;i++){
    fprintf(outfile,"%-10.6g ",g_pot.opt_pot.table[g_pot.opt_pot.idx[i]]);
    }
  fprintf(outfile,"%-10.6g 1          0          -          -\n", cost_after);
#endif


  /* If move not accepted, return 0 */
  return 0;
}

#endif  /* UQ&&APOT */
