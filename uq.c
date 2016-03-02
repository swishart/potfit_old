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

int main (int argc, char *argv[]) { 

	int i;

	char *filename = "tester";
	char *samplefile = "tester_1";

	printf("Starting sample file is: %s\n",samplefile);

	for (i=0;i<10;i++) {
		increment_filename(filename, samplefile);
		printf("Samplefile in main = %s\n",samplefile);
		printf("i = %d\n",i);
	}

	printf("End of program\n");

	return 0;

}

/****************************************************************
 *
 * increment sample output filename
 *
 ****************************************************************/
void increment_filename(char *filename, char *sample_file) {

 	/* Check if tempfile has a number at the end of its name */
 	char *increment;
 	int increment_num;

 	printf("Internal sample file is: %s\n",sample_file);

 	increment = strtok (sample_file, filename);

 	printf("increment is: %s\n", increment);

 	/* If strings are the same, return sample_file_0 */ 
 	if (increment == NULL) {


 		printf ("HELLO\n");
 		//strcat(sample_file, "_0");

 		printf("New sample file is: %s\n",sample_file);

 		return;
 	}
 	printf("HI\n");
	/* Remove underscore from start of increment */ 
	increment = strtok (increment, "_");

    increment_num = atoi(increment);

    printf("integer value of the string is %d\n", increment_num);

    increment_num += 1;
    increment = (char) increment_num;

    printf("new value of the string is %s\n", increment);

    /* add new increment to sample_file */
   // strcat(sample_file, "_");
    strcat(sample_file, increment);

    printf("New sample file is: %s\n",sample_file);

  	return;

 }