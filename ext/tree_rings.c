// #include <ruby.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>

#define DEFAULT_RINGS 1000000000
#define DEBUG 0

#define NUM_THREADS 8


/*
From: http://www.gnu.org/software/libtool/manual/libc/Elapsed-Time.html
Subtract the `struct timeval' values X and Y, storing the result in RESULT.
Return 1 if the difference is negative, otherwise 0.  */     
int timeval_subtract (result, x, y)
     struct timeval *result, *x, *y;
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (x->tv_usec - y->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_usec = x->tv_usec - y->tv_usec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}


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
	
	
	struct timeval start, stop, diff;	
	
	printf("Running serial calculation using CPU...\t\t\t");
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
	

	printf("Done!");	
	return EXIT_SUCCESS;
}

