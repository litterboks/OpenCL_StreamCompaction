#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

class BlellochScan
{
	// defines
	static const int	MAX_NUM_PLATFORMS = 64;
	static const int	MAX_LENGTH_PLATFORM_NAME = 128;
	static const int	BLOCK_SIZE = 2;
	static const char*	kernelPath;

public:
	BlellochScan(unsigned int nPlatform);
	~BlellochScan();

	void RunBlellochScan(int *& inputData, int *& outputData, const unsigned int & nArraySize);
	unsigned int GetError();

private:
	//methods
	void RecursiveScan(const cl_mem& inputBufferAddress, cl_mem& outputBufferAddress, const unsigned int & nArraySize, cl_mem sumsBuffer, unsigned int&  sumsBufferSize);
	void SimpleScan(const cl_mem& inputBufferAddress, cl_mem& outputBufferAddress, const unsigned int & nArraySize, cl_mem& sumsBuffer, unsigned int& nSumsSize);
	void AddSums(const unsigned int & nArraySize, cl_mem outputBufferAddress, cl_mem sumsBuffer);

	//helpers
	bool ReadProgramFromFile(const char* path, char* &source_str);
	void PrintPlatformInformation(const cl_uint &numPlatforms, cl_int &ciErrNum, cl_platform_id  platforms[64]);
	void PrintKernelFunctionName(cl_int &ciErrNum, const cl_kernel &kernel);

	//members
	cl_int m_ciErrNum;										// error number
	cl_device_id m_device;									// device id
	cl_platform_id* m_platforms;							// array for platforms
	cl_uint numPlatforms;									// number of platforms
	cl_context m_context;									// openCL context
	cl_command_queue m_commandQueue;						// the command queue
	cl_kernel m_scanKernel;									// kernel for blelloch scan
	cl_kernel m_addKernel;									// kernel for adding sums
};