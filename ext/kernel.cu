#include <cuda.h>

__global__ void kernel(int N, float *g_result){
  	int n = threadIdx.x + blockDim.x * blockIdx.x;
  	/* compute area - rings start at 1 */
    g_result[n] = ((float)M_PI*(n+1)*(n+1)) - ((float)M_PI*n*n);
}

/* only use extern if calling code is C */
extern "C"
{
	/* driver for kernel */
	void cudakernel(int N, float *g_result){
  		/* choose 256 threads per block for high occupancy */
  		int ThreadsPerBlock = 256;
  		/* find number of blocks */
  		int BlocksPerGrid = (N+ThreadsPerBlock-1)/ThreadsPerBlock;
  		/* invoke device on this block/thread grid */
		kernel <<< BlocksPerGrid, ThreadsPerBlock >>> (N, g_result);
	}
}