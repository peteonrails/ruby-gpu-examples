/*
   A baseline CPU-based benchmark program CPU/GPU performance comparision.
   Copyright © 2011 Preston Lee. All rights reserved.
   http://prestonlee.com 
*/

#ifndef TREE_RINGS
#define TREE_RINGS

typedef struct {
	int rings;
	int number;
} ring_thread_data;

void calculate_ring_areas_in_serial(int rings);
void calculate_ring_areas_in_serial_with_offset(int rings, int thread);
void calculate_ring_areas_in_parallel(int rings);
void calculate_ring_areas_on_GPU();
void ring_job(ring_thread_data * data);

/* From: http://www.gnu.org/software/libtool/manual/libc/Elapsed-Time.html
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

#endif