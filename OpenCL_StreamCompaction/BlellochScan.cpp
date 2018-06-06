#include <iostream>
#include <cmath>

#include "BlellochScan.h"

const char* BlellochScan::kernelPath = "kernel.cl";

BlellochScan::BlellochScan(unsigned int nPlatform)
{
	m_platforms = new cl_platform_id[MAX_NUM_PLATFORMS];
	m_ciErrNum = clGetPlatformIDs(MAX_NUM_PLATFORMS, m_platforms, &numPlatforms);								// get the platforms
	if (nPlatform == 0)
	{
		m_ciErrNum += clGetDeviceIDs(m_platforms[nPlatform], CL_DEVICE_TYPE_CPU, 1, &m_device, NULL);				// get devices on platform nPlatform
	}
	else if(nPlatform == 1)
	{
		m_ciErrNum += clGetDeviceIDs(m_platforms[nPlatform], CL_DEVICE_TYPE_GPU, 1, &m_device, NULL);				// get devices on platform nPlatform
	}
	else
	{
		m_ciErrNum = -2000;
	}

	PrintPlatformInformation(numPlatforms, m_ciErrNum, m_platforms);

	int tempErrorNum;

	m_context = clCreateContext(NULL, 1, &m_device, NULL, NULL, NULL);											//create a context
	m_commandQueue = clCreateCommandQueue(m_context, m_device, (cl_command_queue_properties)0, &tempErrorNum);	//create a command queue
	m_ciErrNum += tempErrorNum;

	char* source_str = nullptr ;
	bool bFileReadSuccess = ReadProgramFromFile(kernelPath, source_str);

	if (!bFileReadSuccess)
	{
		m_ciErrNum = -4000;
	}

	cl_program m_program = clCreateProgramWithSource(m_context, 1, (const char**)&source_str, NULL, &tempErrorNum);	// create the program
	m_ciErrNum += tempErrorNum;
	m_ciErrNum += clBuildProgram(m_program, 0, NULL, NULL, NULL, NULL);												// build the program

	// create the kernels
	m_scanKernel = clCreateKernel(m_program, "scan_init", &tempErrorNum);
	m_addKernel = clCreateKernel(m_program, "add_sums", &tempErrorNum);
	m_ciErrNum += tempErrorNum;
}

BlellochScan::~BlellochScan()
{
	delete[] m_platforms;
}

void BlellochScan::RunBlellochScan(int *& inputData, int *& outputData, const unsigned int & nArraySize)
{
	// create a buffer for the initial input
	cl_mem inputBuffer = clCreateBuffer(m_context, CL_MEM_READ_WRITE, nArraySize * sizeof(cl_int), NULL, NULL);
	clEnqueueWriteBuffer(m_commandQueue, inputBuffer, CL_TRUE, 0, nArraySize * sizeof(cl_int), inputData, 0, NULL, NULL);

	//address of the output buffer will be determined in the recursive blelloch
	cl_mem outputBuffer;

	//size of the sumsBuffer
	unsigned int nSumsSize = 0;
	RecursiveScan(inputBuffer, outputBuffer, nArraySize, 0, nSumsSize);

	clEnqueueReadBuffer(m_commandQueue, outputBuffer, CL_TRUE, 0, nArraySize * sizeof(cl_int), outputData, 0, NULL, NULL);
	clFinish(m_commandQueue);
}

// check if something went wrong
unsigned int BlellochScan::GetError()
{
	return (unsigned int)m_ciErrNum;
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
	if (sumsBufferSize > BLOCK_SIZE)
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
	size_t localws[] = { (size_t)std::fmin(*globalws, BLOCK_SIZE / 2) };

	// create buffers for the data
	nSumsSize = (unsigned int)(*globalws / *localws);
	sumsBuffer = clCreateBuffer(m_context, CL_MEM_READ_WRITE, sizeof(int) * nSumsSize, NULL, NULL);
	outputBufferAddress = clCreateBuffer(m_context, CL_MEM_READ_WRITE, sizeof(int) * nArraySize, NULL, NULL);

	// set the arguments
	clSetKernelArg(m_scanKernel, 0, sizeof(cl_mem), &inputBufferAddress);
	clSetKernelArg(m_scanKernel, 1, sizeof(cl_mem), &outputBufferAddress);
	clSetKernelArg(m_scanKernel, 2, sizeof(int) * (*localws) * 2, NULL);
	clSetKernelArg(m_scanKernel, 3, sizeof(cl_mem), &sumsBuffer);

	//execute kernel
	clEnqueueNDRangeKernel(m_commandQueue, m_scanKernel, 1, 0, globalws, localws, 0, NULL, NULL);
}

void BlellochScan::AddSums(const unsigned int & nArraySize, cl_mem outputBufferAddress, cl_mem sumsBuffer)
{
	// workgroup sizes
	size_t globalws[] = { (size_t)std::fmax(nArraySize / 2, 1) };
	size_t localws[] = { (size_t)std::fmin(*globalws, BLOCK_SIZE / 2) };

	// set the arguments
	clSetKernelArg(m_addKernel, 0, sizeof(cl_mem), &outputBufferAddress);
	clSetKernelArg(m_addKernel, 1, sizeof(cl_mem), &sumsBuffer);

	// execute the kernel
	clEnqueueNDRangeKernel(m_commandQueue, m_addKernel, 1, 0, globalws, localws, 0, NULL, NULL);
}

// read the program from a file
inline bool BlellochScan::ReadProgramFromFile(const char* path, char *& source_str)
{
	FILE *fp;
	size_t program_size;

	fopen_s(&fp, "scan.cl", "rb");
	if (!fp) {
		printf("Failed to load kernel\n");
		return 0;
	}

	fseek(fp, 0, SEEK_END);
	program_size = ftell(fp);
	rewind(fp);
	source_str = (char*)malloc(program_size + 1);
	source_str[program_size] = '\0';
	fread(source_str, sizeof(char), program_size, fp);
	fclose(fp);
	return 1;
}

// print platform information for debugging purposes
inline void BlellochScan::PrintPlatformInformation(const cl_uint & numPlatforms, cl_int & ciErrNum, cl_platform_id platforms[64])
{
	char platformName[MAX_LENGTH_PLATFORM_NAME];				// destination array for platform name
	size_t sizeRetPlatformName;									// actual length of platform name

	// print a list of all platforms
	printf("Platforms:\n");
	for (unsigned int i = 0; i < numPlatforms; ++i)
	{
		ciErrNum = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(char) * 120, platformName, &sizeRetPlatformName);
		printf("Plaform %d: ", i);
		printf(platformName);
		printf("\n");
	}
	printf("\n");
}

inline void BlellochScan::PrintKernelFunctionName(cl_int & ciErrNum, const cl_kernel & kernel)
{
	char kernelFunctionName[120];
	size_t sizeRetKernelFunctionName; // actual length of the kernel function name
	ciErrNum = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 120, kernelFunctionName, &sizeRetKernelFunctionName);
	printf("\n");
	printf(kernelFunctionName);
	printf("\n");
}
