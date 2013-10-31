/****************************************************************
 *
 * forces.c: General settings for force routines
 *
 ****************************************************************
 *
 * Copyright 2002-2013
 *	Institute for Theoretical and Applied Physics
 *	University of Stuttgart, D-70550 Stuttgart, Germany
 *	http://potfit.sourceforge.net/
 *
 ****************************************************************
 *
 *   This file is part of potfit.
 *
 *   potfit is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   potfit is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with potfit; if not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************/

#include "potfit.h"

/****************************************************************
 *
 *  set_forces() assigns the correct pointer to the
 *  	calc_forces function pointer.
 *
 ****************************************************************/

void set_forces()
{

  /* assign correct force routine */
#ifdef PAIR
  calc_forces = calc_forces_pair;
#elif defined EAM && !defined COULOMB
  calc_forces = calc_forces_eam;
#elif defined ADP
  calc_forces = calc_forces_adp;
#elif defined COULOMB && !defined EAM
  calc_forces = calc_forces_elstat;
#elif defined COULOMB && defined EAM
  calc_forces = calc_forces_eam_elstat;
#elif defined MEAM
  calc_forces = calc_forces_meam;
#elif defined STIWEB
  calc_forces = calc_forces_stiweb;
#elif defined TERSOFF
  calc_forces = calc_forces_tersoff;
#endif /* PAIR */

  return;
}

/****************************************************************
 *
 *  init_forces() is called after all parameters and potentials are
 *  	read. Additional assignments and initializations can be made here.
 *
 ****************************************************************/

void init_forces()
{
  int   i;

#ifdef COULOMB
  if (apot_table.sw_kappa)
    init_tails(apot_table.dp_kappa[0]);
#endif /* COULOMB */

  /* set spline density corrections to 0 */
#if defined EAM || defined ADP || defined MEAM
  lambda = (double *)malloc(ntypes * sizeof(double));
  reg_for_free(lambda, "lambda");
  for (i = 0; i < ntypes; i++)
    lambda[i] = 0.0;
#endif /* EAM || ADP || MEAM */

  return;
}

/****************************************************************
 *
 *  set_force_vector_pointers() assigns the correct positions to the
 *  	pointers for the force vector
 *
 *  	energy_p 	... 	position of the first energy value
 *  	stress_p 	... 	position of the first stress value
 *  	limit_p 	... 	position of the first limiting constraint
 *  	dummy_p 	... 	position of the first dummy constraint
 *  	punish_par_p 	... 	position of the first parameter punishment (APOT only)
 *  	punish_pot_p 	... 	position of the first potential punishment (APOT only)
 *
 ****************************************************************/

void set_force_vector_pointers()
{
  energy_p = 3 * natoms;

#ifdef STRESS
  stress_p = energy_p + nconf;
#if defined EAM || defined ADP || defined MEAM
  limit_p = stress_p + 6 * nconf;
  dummy_p = limit_p + nconf;
#ifdef APOT
  punish_par_p = dummy_p + 2 * ntypes;
  punish_pot_p = punish_par_p + apot_table.total_par - apot_table.invar_pots;
#endif /* APOT */
#else /* EAM || ADP || MEAM */
#ifdef APOT
  punish_par_p = stress_p + 6 * nconf;
  punish_pot_p = punish_par_p + apot_table.total_par;
#endif /* APOT */
#endif /* EAM || ADP || MEAM */

#else /* STRESS */

#if defined EAM || defined ADP || defined MEAM
  limit_p = energy_p + nconf;
  dummy_p = limit_p + nconf;
#ifdef APOT
  punish_par_p = dummy_p + 2 * ntypes;
  punish_pot_p = punish_par_p + apot_table.total_par - apot_table.invar_pots;
#endif /* APOT */
#else /* EAM || ADP || MEAM */
#ifdef APOT
  punish_par_p = energy_p + nconf;
  punish_pot_p = punish_par_p + apot_table.total_par;
#endif /* APOT */
#endif /* EAM || ADP || MEAM */

#endif /* STRESS */

  return;
}