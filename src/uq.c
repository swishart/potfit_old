/****************************************************************
 *
 * uq.c: Routines for uncertainty quantification
 *
 ****************************************************************
 *
 * Copyright 2002-2014
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

//#include "potfit.h"
//#include "potential.h"
 #include <stdio.h>
 #include <string.h>
 #include <ctype.h>
 #include <stdlib.h>

char increment_filename(char *, char *);

int main (int argc, char *argv[]) { 

	int j;

	char filename[100] = "potfit_Si";
    char output[100];

	for (j=0;j<10;j++) {
		increment_filename(filename, output);
	}

	return 0;
}

/****************************************************************
 *
 * increment sample output filename
 *
 ****************************************************************/
char increment_filename(char *filename, char *output) {

 	/* Check if tempfile has a number at the end of its name */
	static int i = 0;

 	char increment[100];

    sprintf(increment, "_%d", i);
    sprintf(output, "%s%s", filename, increment);
    
    i++;

  	return;

 }