// #include <ruby.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define DEFAULT_RINGS 800000000
#define DEBUG 0

#define NUM_THREADS 8



// Returns the execution time in milliseconds.
void calculate_ring_areas_in_serial_with_offset(int rings, int thread) {
	int i;
	int offset = rings * thread;
	int max = rings + offset;
	float a = 0;
	for(i = offset; i < max; i++) {
		a = (M_PI * pow(i, 2)) - (M_PI * pow(i - 1, 2));
		// a = a + i * a + i;
		// printf("T %i area: %f\n", thread, a);
	}
}

void calculate_ring_areas_in_serial(int rings) {
	calculate_ring_areas_in_serial_with_offset(rings, 0);
}


typedef struct {
	int rings;
	int number;
} ring_thread_data;

void ring_job(ring_thread_data * data) {
	// data->rings;
	calculate_ring_areas_in_serial_with_offset(data->rings, data->number);
}

void calculate_ring_areas_in_parallel(int rings) {	
	pthread_t threads[NUM_THREADS];
	int rc;
	int t;
	int rings_per_thread = rings / NUM_THREADS;
	ring_thread_data data[NUM_THREADS];
	
	for(t = 0; t < NUM_THREADS; t++){
	  // printf("In main: creating thread %i\n", t);
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

int main(int argc, const char * argv[]) {
	int rings = DEFAULT_RINGS;
	// int rings = 100;
	if(argc > 1) {
		rings = atoi(argv[1]);
	}
	
	
	printf("A baseline CPU-based benchmark program CPU/GPU performance comparision.\n");
	printf("Copyright © 2011 Preston Lee. All rights reserved.\n\n");
	printf("\tUsage: %s [NUM TREE RINGS]\n\n", argv[0]);
	
	printf("Number of tree rings: %i. Yay!\n", rings);
	
	
	clock_t start, stop;
	double cpu_time_used;
	
	printf("Running serial calculation using CPU...\t\t\t");
	start = clock();
	calculate_ring_areas_in_serial(rings);
	stop = clock();
	cpu_time_used = ((double) (stop - start)) * 1000.0 / CLOCKS_PER_SEC;
	printf("Time: %lf milliseconds.\n", cpu_time_used);
	
	
	printf("Running parallel calculation using %i CPU threads...\t", NUM_THREADS);
	start = clock();
	calculate_ring_areas_in_parallel(rings);
	stop = clock();
	cpu_time_used = ((double) (stop - start)) * 1000.0 / CLOCKS_PER_SEC;
	printf("Time: %lf milliseconds.\n", cpu_time_used);

	//printf("Current time: %d", start.time);

	printf("Done!");	
	return EXIT_SUCCESS;
}

