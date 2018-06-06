#include<iostream>

#define _USE_MATH_DEFINES
#include <math.h>
#include "BlellochScan.h"

//defines for my system
#define INTEL_CPU  0
#define NVIDIA_GPU 1

int main(void)
{
	BlellochScan blellochScan(NVIDIA_GPU);

	// something went wrong
	if (blellochScan.GetError() != 0)
	{
		return -1;
	}


	unsigned int nArraySize = 2048;
	int* inputData = new int[nArraySize];

	for (unsigned int i = 0; i < nArraySize; i++)
	{
		inputData[i] = 1;
	}

	int* outputData = new int[nArraySize];
	
	blellochScan.RunBlellochScan(inputData, outputData, nArraySize);

	return 0;
}



