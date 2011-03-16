/*
   A baseline CPU-based benchmark program CPU/GPU performance comparision.
   Approximates the cross-sectional area of every tree ring in a tree trunk in serial and in parallel
   by taking the total area at a given radius and subtracting the area of the closest inner ring.
   Copyright © 2011 Preston Lee. All rights reserved.
   http://prestonlee.com 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "tree_rings.h"

#define DEFAULT_RINGS pow(2,25)	//currently only works on powers of 2
#define NUM_THREADS 8
#define DEBUG 0

//float res = 0;

int main(int argc, const char * argv[]) {
	int rings = DEFAULT_RINGS;

	if(argc > 1) {
		rings = atoi(argv[1]);
	}
	
	printf("\nA baseline CPU-based benchmark program for CPU/GPU performance comparision.\n");
	printf("Copyright © 2011 Preston Lee. All rights reserved.\n\n");
	printf("\tUsage: %s [NUM TREE RINGS]\n\n", argv[0]);
	
	printf("Number of tree rings: %i. Yay!\n", rings);
		
	struct timeval start, stop, diff;	

	//printf("\nArea 0 = %f\n", M_PI*pow(DEFAULT_RINGS, 2));
		
	printf("\nRunning serial calculation using CPU...\t\t\t");
	gettimeofday(&start, NULL);
	calculate_ring_areas_in_serial(rings);
	gettimeofday(&stop, NULL);
	timeval_subtract(&diff, &stop, &start);
	printf("%ld.%06ld seconds\n", (long)diff.tv_sec, (long)diff.tv_usec);
	//printf("Area 1 = %f\n", res);
	//res = 0;
	
	printf("Running parallel calculation using %i CPU threads...\t", NUM_THREADS);
	gettimeofday(&start, NULL);
	calculate_ring_areas_in_parallel(rings);
	gettimeofday(&stop, NULL);
	timeval_subtract(&diff, &stop, &start);
	printf("%ld.%06ld seconds\n", (long)diff.tv_sec, (long)diff.tv_usec);
	//printf("Area 2 = %f\n", res);
	//res = 0;

	printf("Running parallel calculation using GPU...\t\t");
	gettimeofday(&start, NULL);
	calculate_ring_areas_on_GPU();
	gettimeofday(&stop, NULL);
	timeval_subtract(&diff, &stop, &start);
	printf("%ld.%06ld seconds\n", (long)diff.tv_sec, (long)diff.tv_usec);
	//printf("Area 3 = %f\n", res);
	
	printf("\nDone!\n\n");	
	return EXIT_SUCCESS;
}

/*  Approximate the cross-sectional area between each pair of consecutive tree rings
 	in serial */
void calculate_ring_areas_in_serial(int rings) {
		calculate_ring_areas_in_serial_with_offset(rings, 0);
}

void calculate_ring_areas_in_serial_with_offset(int rings, int thread) {
	int i;
	int offset = rings * thread;
	int max = rings + offset;
	float a = 0;
	for(i = offset+1; i < max+1; i++) {
		a = (M_PI * pow(i, 2)) - (M_PI * pow(i - 1, 2));
		//res += a;
	}
}

/*  Approximate the cross-sectional area between each pair of consecutive tree rings
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
	    rc = pthread_create(&threads[t], NULL, (void *) ring_job, (void *) &data[t]);
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

/*  Approximate the cross-sectional area between each pair of consecutive tree rings
 	in parallel on a GPU */
void calculate_ring_areas_on_GPU(){
	/* registers */
  	int n;
	/* device ID */
	int devid;
  	/* device count */
	int devcount;
	/* number of entries in arrays */
	int N = DEFAULT_RINGS;
	/* pointer to host array */
	float *h_array;
	/* pointer to gpu device array */
	float *g_array;

 	/* find number of device in current "context" */
	cudaGetDevice(&devid);

	/* find how many devices are available */
	cudaGetDeviceCount(&devcount);

	/* allocate array on host (via CUDA) */
	cudaMallocHost((void**) &h_array, N*sizeof(float));

	/* allocate arrays on device (via CUDA) */
	cudaMalloc((void**) &g_array, N*sizeof(float));

	/* invoke kernel on device */
	void cudakernel(int N, float *g_result);
	cudakernel(N, g_array);

	/* copy from device array to host array */
	cudaMemcpy(h_array, g_array, N*sizeof(float), cudaMemcpyDeviceToHost);
	
	//int i;
	//for(i=0;i<N;i++){
	//	res += h_array[i];
	//}		
}