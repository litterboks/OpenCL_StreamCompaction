#include<iostream>

#define _USE_MATH_DEFINES
#include <math.h>
#include "StreamCompaction.h"
#include "GpgpuSetup.h"
#include "BlellochScan.h"

//defines for my system
#define INTEL_CPU  0
#define NVIDIA_GPU 1

int main(void)
{
	GpgpuSetup gpgpuSetup(NVIDIA_GPU);
	BlellochScan blellochScan(&gpgpuSetup);
	StreamCompaction streamCompaction(&gpgpuSetup);

	// something went wrong
	if (streamCompaction.GetError() != 0)
	{
		return -1;
	}

	unsigned int nArraySize = 536870912;
	int* inputData = new int[nArraySize];

	for (unsigned int i = 0; i < nArraySize; i++)
	{
		inputData[i] = i;
	}

	int* outputData;// = new int[nArraySize];

	int resultSize = streamCompaction.CompactStream(inputData, outputData, nArraySize, StreamCompaction::Predicate::ODD);

	return 0;
}