#include <iostream>
#include "GpgpuSetup.h"

const char* GpgpuSetup::kernelPath = "kernel.cl";

GpgpuSetup::GpgpuSetup(unsigned int nPlatform)
{
	m_platforms = new cl_platform_id[MAX_NUM_PLATFORMS];
	m_ciErrNum = clGetPlatformIDs(MAX_NUM_PLATFORMS, m_platforms, &m_numPlatforms);									// get the platforms
	if (nPlatform == 0)
	{
		m_ciErrNum += clGetDeviceIDs(m_platforms[nPlatform], CL_DEVICE_TYPE_CPU, 1, &m_device, NULL);				// get devices on platform nPlatform
	}
	else if (nPlatform == 1)
	{
		m_ciErrNum += clGetDeviceIDs(m_platforms[nPlatform], CL_DEVICE_TYPE_GPU, 1, &m_device, NULL);				// get devices on platform nPlatform
	}
	else
	{
		m_ciErrNum = -2000;
	}

	PrintPlatformInformation(m_numPlatforms, m_ciErrNum, m_platforms);

	int tempErrorNum;

	char* source_str = nullptr;
	bool bFileReadSuccess = GpgpuSetup::ReadProgramFromFile(kernelPath, source_str);

	if (!bFileReadSuccess)
	{
		m_ciErrNum = -4000;
	}

	m_context = clCreateContext(NULL, 1, &m_device, NULL, NULL, NULL);												//create a context
	m_commandQueue = clCreateCommandQueue(m_context, m_device, (cl_command_queue_properties)0, &tempErrorNum);		//create a command queue
	m_ciErrNum += tempErrorNum;

	m_program = clCreateProgramWithSource(m_context, 1, (const char**)&source_str, NULL, &tempErrorNum);			// create the program
	m_ciErrNum += tempErrorNum;
	m_ciErrNum += clBuildProgram(m_program, 0, NULL, NULL, NULL, NULL);												// build the program
}

GpgpuSetup::GpgpuSetup(GpgpuSetup * openCLSetup)
{
	m_ciErrNum = openCLSetup->m_ciErrNum;
	m_device = openCLSetup->m_device;
	m_numPlatforms = openCLSetup->m_numPlatforms;

	m_platforms = new cl_platform_id[m_numPlatforms];
	memcpy(m_platforms, openCLSetup->m_platforms, sizeof(cl_platform_id)* m_numPlatforms);

	m_context = openCLSetup->m_context;
	m_commandQueue = openCLSetup->m_commandQueue;
	m_program = openCLSetup->m_program;
}

GpgpuSetup::~GpgpuSetup()
{
	delete[] m_platforms;
}

bool GpgpuSetup::ReadProgramFromFile(const char * path, char *& source_str)
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

void GpgpuSetup::PrintPlatformInformation(const cl_uint & numPlatforms, cl_int & ciErrNum, cl_platform_id platforms[64])
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

void GpgpuSetup::PrintKernelFunctionName(cl_int & ciErrNum, const cl_kernel & kernel)
{
	char kernelFunctionName[120];
	size_t sizeRetKernelFunctionName; // actual length of the kernel function name
	ciErrNum = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 120, kernelFunctionName, &sizeRetKernelFunctionName);
	printf("\n");
	printf(kernelFunctionName);
	printf("\n");
}
