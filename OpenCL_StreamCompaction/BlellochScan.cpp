#include <iostream>
#include <cmath>

#include "BlellochScan.h"
#include "GpgpuSetup.h"

struct BlellochScan::Kernels {
	cl_kernel scanKernel;		// kernel for blelloch scan
	cl_kernel addKernel;		// kernel for adding sums
};

BlellochScan::BlellochScan(unsigned int nPlatform)
{
	m_gpgpuSetup = new GpgpuSetup(nPlatform);
	CreateKernels();
}

BlellochScan::BlellochScan(GpgpuSetup * gpgpuSetup)
{
	m_gpgpuSetup = new GpgpuSetup(gpgpuSetup);
	CreateKernels();
}

void BlellochScan::CreateKernels()
{
	// create the kernels
	int tempErrorNum = 0;
	this->m_kernels = new BlellochScan::Kernels();
	m_kernels->scanKernel = clCreateKernel(m_gpgpuSetup->m_program, "scan_init", &tempErrorNum);
	m_gpgpuSetup->m_ciErrNum += tempErrorNum;
	m_kernels->addKernel = clCreateKernel(m_gpgpuSetup->m_program, "add_sums", &tempErrorNum);
	m_gpgpuSetup->m_ciErrNum += tempErrorNum;
}

BlellochScan::~BlellochScan()
{
	delete m_kernels;
	delete m_gpgpuSetup;
}

void BlellochScan::RunBlellochScan(int *& inputData, int *& outputData, const unsigned int & nArraySize)
{
	// create a buffer for the initial input
	cl_mem inputBuffer = clCreateBuffer(m_gpgpuSetup->m_context, CL_MEM_READ_WRITE, nArraySize * sizeof(cl_int), NULL, NULL);
	clEnqueueWriteBuffer(m_gpgpuSetup->m_commandQueue, inputBuffer, CL_TRUE, 0, nArraySize * sizeof(cl_int), inputData, 0, NULL, NULL);

	//address of the output buffer will be determined in the recursive blelloch
	cl_mem outputBuffer;

	//size of the sumsBuffer
	unsigned int nSumsSize = 0;
	RecursiveScan(inputBuffer, outputBuffer, nArraySize, 0, nSumsSize);

	clEnqueueReadBuffer(m_gpgpuSetup->m_commandQueue, outputBuffer, CL_TRUE, 0, nArraySize * sizeof(cl_int), outputData, 0, NULL, NULL);
	clFinish(m_gpgpuSetup->m_commandQueue);
}

void BlellochScan::RunBlellochScan(const cl_mem& inputBuffer, cl_mem& outputBuffer, const unsigned int & nArraySize)
{
	//size of the sumsBuffer
	unsigned int nSumsSize = 0;
	RecursiveScan(inputBuffer, outputBuffer, nArraySize, 0, nSumsSize);
}

// check if something went wrong
unsigned int BlellochScan::GetError()
{
	return m_gpgpuSetup->m_ciErrNum;
}

void BlellochScan::RecursiveScan(const cl_mem& inputBufferAddress, cl_mem& outputBufferAddress, const unsigned int & nArraySize, cl_mem sumsBufferAddress, unsigned int&  sumsBufferSize)
{
	SimpleScan(inputBufferAddress, outputBufferAddress, nArraySize, sumsBufferAddress, sumsBufferSize);
	
	//sums buffer for recursive steps
	cl_mem sumsBuffer = 0;
	unsigned int nSumsSize = 0;

	// output of the recursive steps
	cl_mem sumsOutput;

	// entweder größer als block size, dann rekursiv aufrufen oder kleiner gleich block size dann kann abgearbeitet werden
	if (sumsBufferSize > GpgpuSetup::BLOCK_SIZE)
	{
		RecursiveScan(sumsBufferAddress, sumsOutput, sumsBufferSize, sumsBuffer, nSumsSize);
	}
	else
	{
		SimpleScan(sumsBufferAddress, sumsOutput, sumsBufferSize, sumsBuffer, nSumsSize);
	}
	sumsBufferAddress = sumsOutput;

	// summing up the recursive steps
	AddSums(nArraySize, outputBufferAddress, sumsBufferAddress);

	// release memory
	clReleaseMemObject(sumsBuffer);
}

void BlellochScan::SimpleScan(const cl_mem& inputBufferAddress, cl_mem& outputBufferAddress, const unsigned int & nArraySize, cl_mem& sumsBuffer, unsigned int& nSumsSize)
{
	//workgroup sizes
	size_t globalws[] = { (size_t)std::fmax(nArraySize / 2, 1) };
	size_t localws[] = { (size_t)std::fmin(*globalws, GpgpuSetup::BLOCK_SIZE / 2) };

	// create buffers for the data
	nSumsSize = (unsigned int)(*globalws / *localws);
	sumsBuffer = clCreateBuffer(m_gpgpuSetup->m_context, CL_MEM_READ_WRITE, sizeof(int) * nSumsSize, NULL, NULL);
	outputBufferAddress = clCreateBuffer(m_gpgpuSetup->m_context, CL_MEM_READ_WRITE, sizeof(int) * nArraySize, NULL, NULL);

	// set the arguments
	clSetKernelArg(m_kernels->scanKernel, 0, sizeof(cl_mem), &inputBufferAddress);
	clSetKernelArg(m_kernels->scanKernel, 1, sizeof(cl_mem), &outputBufferAddress);
	clSetKernelArg(m_kernels->scanKernel, 2, sizeof(int) * (*localws) * 2, NULL);
	clSetKernelArg(m_kernels->scanKernel, 3, sizeof(cl_mem), &sumsBuffer);

	//execute kernel
	clEnqueueNDRangeKernel(m_gpgpuSetup->m_commandQueue, m_kernels->scanKernel, 1, 0, globalws, localws, 0, NULL, NULL);
}

void BlellochScan::AddSums(const unsigned int & nArraySize, cl_mem outputBufferAddress, cl_mem sumsBuffer)
{
	// workgroup sizes
	size_t globalws[] = { (size_t)std::fmax(nArraySize / 2, 1) };
	size_t localws[] = { (size_t)std::fmin(*globalws, GpgpuSetup::BLOCK_SIZE / 2) };

	// set the arguments
	clSetKernelArg(m_kernels->addKernel, 0, sizeof(cl_mem), &outputBufferAddress);
	clSetKernelArg(m_kernels->addKernel, 1, sizeof(cl_mem), &sumsBuffer);

	// execute the kernel
	clEnqueueNDRangeKernel(m_gpgpuSetup->m_commandQueue, m_kernels->addKernel, 1, 0, globalws, localws, 0, NULL, NULL);
}
