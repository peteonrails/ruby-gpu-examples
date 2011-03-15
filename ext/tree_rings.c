/*
   A baseline CPU-based benchmark program CPU/GPU performance comparision.
   Copyright © 2011 Preston Lee. All rights reserved.
   http://prestonlee.com 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

#include "tree_rings.h"

#define DEFAULT_RINGS 1000000000
#define NUM_THREADS 8
#define DEBUG 0

/*  Calculates the area between each pair of consecutive rings in a tree
 	in serial, and then in parallel on NUM_THREADS threads */
int main(int argc, const char * argv[]) {
	int rings = DEFAULT_RINGS;

	if(argc > 1) {
		rings = atoi(argv[1]);
	}
	
	printf("A baseline CPU-based benchmark program for CPU/GPU performance comparision.\n");
	printf("Copyright © 2011 Preston Lee. All rights reserved.\n\n");
	printf("\tUsage: %s [NUM TREE RINGS]\n\n", argv[0]);
	
	printf("Number of tree rings: %i. Yay!\n", rings);
		
	struct timeval start, stop, diff;	
	
	printf("\nRunning serial calculation using CPU...\t\t\t");
	gettimeofday(&start, NULL);
	calculate_ring_areas_in_serial(rings);
	gettimeofday(&stop, NULL);
	timeval_subtract(&diff, &stop, &start);
	printf(" (%ld seconds, %ld microseconds)\n", (long)diff.tv_sec, (long)diff.tv_usec);
	
	printf("Running parallel calculation using %i CPU threads...\t", NUM_THREADS);
	gettimeofday(&start, NULL);
	calculate_ring_areas_in_parallel(rings);
	gettimeofday(&stop, NULL);
	timeval_subtract(&diff, &stop, &start);
	printf(" (%ld seconds, %ld microseconds)\n", (long)diff.tv_sec, (long)diff.tv_usec);
	

	printf("\nDone!");	
	return EXIT_SUCCESS;
}

/*  Calculates the area between each pair of consecutive rings in a tree
 	in serial */
void calculate_ring_areas_in_serial(int rings) {
		calculate_ring_areas_in_serial_with_offset(rings, 0);
}

void calculate_ring_areas_in_serial_with_offset(int rings, int thread) {
	int i;
	int offset = rings * thread;
	int max = rings + offset;
	float a = 0;
	for(i = offset; i < max; i++) {
		a = (M_PI * pow(i, 2)) - (M_PI * pow(i - 1, 2));
	}
}

/*  Calculates the area between each pair of consecutive rings in a tree
 	in parallel on NUM_THREADS threads */
void calculate_ring_areas_in_parallel(int rings) {	
	pthread_t threads[NUM_THREADS];
	int rc;
	int t;
	int rings_per_thread = rings / NUM_THREADS;
	ring_thread_data data[NUM_THREADS];
	
	for(t = 0; t < NUM_THREADS; t++){
		data[t].rings = rings_per_thread;
		data[t].number = t;
	    rc = pthread_create(&threads[t], NULL, (void *) ring_job, (void *)&data[t]);
	    if (rc){
	      printf("ERROR; return code from pthread_create() is %d\n", rc);
	      exit(-1);
	   }
	}
	
	for(t = 0; t < NUM_THREADS; t++){
		pthread_join(threads[t], NULL);
	}
}

void ring_job(ring_thread_data * data) {
	calculate_ring_areas_in_serial_with_offset(data->rings, data->number);
}