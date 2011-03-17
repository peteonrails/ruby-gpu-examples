#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE ((unsigned long)pow(2, 40))

int main(const argc, char * argv[]) {
	printf("Allocing %ld bytes", SIZE);
	printf(" (%ldGB)...\n", SIZE / 1024 / 1024 / 1024);
	int * crap = malloc(SIZE);
	if(crap == 0x0) {
		printf("Could not malloc.");
	} else {
		printf("Malloc succeeded!");
	}
	unsigned long i;
	unsigned long max = SIZE;
	
	for(i = 0; i < max; ++i)
	{
		// printf("Writing to %ld", i);
		crap[i] = 42;
	}
	// sleep(100000);
	printf("Freeing...\n");
	printf("%ld", sizeof(crap));
	free(crap);
	printf("Done!\n");
}