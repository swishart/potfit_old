/****************************************************************
 *
 * uq.h:
 *
 ****************************************************************
 *
 * Copyright 2002-2017 - the potfit development team
 *
 * https://www.potfit.net/
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

void uncertainty_quantification(double);

double** calc_hessian(double, int);
int calc_h0_eigenvectors(double**, double, double, double**, double*, int);
int calc_svd(double**, double**, double*, int);
double calc_pot_params(double** const, double** const, double*, double, double*, int*, FILE*);
int mc_moves(double**,double*, double*, int, double, FILE*);

double** mat_double(int, int); /* in powell_lsq.c */
