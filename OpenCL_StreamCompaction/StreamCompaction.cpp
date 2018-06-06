#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "StreamCompaction.h"
#include "GpgpuSetup.h"
#include "BlellochScan.h"
#include <cmath>

struct StreamCompaction::Kernels {
	cl_kernel even;									// kernel for even elements
	cl_kernel odd;									// kernel for odd elements
	cl_kernel lesser500;							// kernel for elements lesser 10000
	cl_kernel compact;
};

StreamCompaction::StreamCompaction(unsigned int nPlatform)
{
	this->m_gpgpuSetup = new GpgpuSetup(nPlatform);
	this->m_blellochScan = new BlellochScan(m_gpgpuSetup);
	CreateKernels();
}

StreamCompaction::StreamCompaction(GpgpuSetup * gpgpuSetup)
{
	this->m_gpgpuSetup = new GpgpuSetup(gpgpuSetup);
	this->m_blellochScan = new BlellochScan(gpgpuSetup);
	CreateKernels();
}

StreamCompaction::~StreamCompaction()
{	
	delete m_gpgpuSetup;
	delete m_kernels;
}

unsigned int StreamCompaction::GetError()
{
	return m_gpgpuSetup->m_ciErrNum;
}

int StreamCompaction::CompactStream(int*& input, int*& output,  unsigned int nArraySize, Predicate predicate)
{
	cl_kernel predicateKernel;
	switch (predicate) {
	case EVEN:
		predicateKernel = m_kernels->even;
		break;
	case ODD:
		predicateKernel = m_kernels->odd;
		break;
	case LESSER500:
		predicateKernel = m_kernels->lesser500;
		break;
	default:
		return -1;
	}

	cl_mem sourceMem = clCreateBuffer(m_gpgpuSetup->m_context, CL_MEM_READ_WRITE, nArraySize * sizeof(int), NULL, &m_gpgpuSetup->m_ciErrNum);
	clEnqueueWriteBuffer(m_gpgpuSetup->m_commandQueue, sourceMem, CL_TRUE, 0, nArraySize * sizeof(int), input, 0, NULL, NULL);

	int* predicateResults = new int[nArraySize];
	cl_mem predicateMem = clCreateBuffer(m_gpgpuSetup->m_context, CL_MEM_READ_WRITE, nArraySize * sizeof(int), NULL, &m_gpgpuSetup->m_ciErrNum);

	clSetKernelArg(predicateKernel, 0, sizeof(cl_mem), &sourceMem);
	clSetKernelArg(predicateKernel, 1, sizeof(cl_mem), &predicateMem);

	size_t globalws[] = { nArraySize };
	size_t localws[] = { nArraySize };

	//execute kernel
	clEnqueueNDRangeKernel(m_gpgpuSetup->m_commandQueue, predicateKernel, 1, 0, globalws, NULL, 0, NULL, NULL);

	clEnqueueReadBuffer(m_gpgpuSetup->m_commandQueue, predicateMem, CL_TRUE, 0, sizeof(int) * nArraySize, predicateResults, 0, NULL, NULL);

	cl_mem addressMem;
	int* addresses = new int[nArraySize];

	m_blellochScan->RunBlellochScan(predicateMem, addressMem, nArraySize);
	clReleaseMemObject(predicateMem);
	clEnqueueReadBuffer(m_gpgpuSetup->m_commandQueue, addressMem, CL_TRUE, 0, sizeof(int) * nArraySize, addresses, 0, NULL, NULL);

	unsigned int highestAddress = addresses[nArraySize -1];
	unsigned int lastPredicateResult = predicateResults[nArraySize - 1];
	int resultSize  = (lastPredicateResult == 0) ? highestAddress : highestAddress + 1;
	cl_mem resultMem = clCreateBuffer(m_gpgpuSetup->m_context, CL_MEM_READ_WRITE, sizeof(int) * resultSize, NULL, &m_gpgpuSetup->m_ciErrNum);

	clSetKernelArg(m_kernels->compact, 0, sizeof(cl_mem), &sourceMem);
	clSetKernelArg(m_kernels->compact, 1, sizeof(cl_mem), &addressMem);
	clSetKernelArg(m_kernels->compact, 2, sizeof(cl_mem), &resultMem);
	clSetKernelArg(m_kernels->compact, 3, sizeof(cl_int), &resultSize);

	//execute kernel
	clEnqueueNDRangeKernel(m_gpgpuSetup->m_commandQueue, m_kernels->compact, 1, 0, globalws, NULL, 0, NULL, NULL);

	clReleaseMemObject(sourceMem);
	clReleaseMemObject(addressMem);

	output = new int[resultSize];
	clEnqueueReadBuffer(m_gpgpuSetup->m_commandQueue, resultMem, CL_TRUE, 0, sizeof(cl_int) * resultSize, output, 0, NULL, NULL);
	clFinish(m_gpgpuSetup->m_commandQueue);

	return resultSize;
}

void StreamCompaction::CreateKernels()
{
	this->m_kernels = new StreamCompaction::Kernels();
	int tempErrorNum = 0;
	m_kernels->even = clCreateKernel(m_gpgpuSetup->m_program, "even", &tempErrorNum);
	m_gpgpuSetup->m_ciErrNum += tempErrorNum;
	m_kernels->odd = clCreateKernel(m_gpgpuSetup->m_program, "odd", &tempErrorNum);
	m_gpgpuSetup->m_ciErrNum += tempErrorNum;
	m_kernels->lesser500 = clCreateKernel(m_gpgpuSetup->m_program, "lesser500", &tempErrorNum);
	m_gpgpuSetup->m_ciErrNum += tempErrorNum;
	m_kernels->compact = clCreateKernel(m_gpgpuSetup->m_program, "compact", &tempErrorNum);
	m_gpgpuSetup->m_ciErrNum += tempErrorNum;
}