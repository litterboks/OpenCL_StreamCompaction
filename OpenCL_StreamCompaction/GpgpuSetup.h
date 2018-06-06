#pragma once
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

// Container Class for OpenCL
class GpgpuSetup
{
public:
	// defines
	static const int	MAX_NUM_PLATFORMS = 64;
	static const int	MAX_LENGTH_PLATFORM_NAME = 128;
	static const int	BLOCK_SIZE = 1024;
	static const char*	kernelPath;

	GpgpuSetup(unsigned int nPlatform);	// constructor
	GpgpuSetup(GpgpuSetup* openCLSetup);	// copy constructor
	~GpgpuSetup();

	//members
	cl_int m_ciErrNum;										// error number
	cl_device_id m_device;									// device id
	cl_platform_id* m_platforms;							// array for platforms
	cl_uint m_numPlatforms;									// number of platforms
	cl_context m_context;									// openCL context
	cl_command_queue m_commandQueue;						// the command queue
	cl_program m_program;								// the program

	static bool ReadProgramFromFile(const char* path, char* &source_str);
	static void PrintPlatformInformation(const cl_uint &numPlatforms, cl_int &ciErrNum, cl_platform_id  platforms[64]);
	static void PrintKernelFunctionName(cl_int &ciErrNum, const cl_kernel &kernel);
};