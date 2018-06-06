class GpgpuSetup;
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

class BlellochScan
{
	struct Kernels;

public:
	//constructor destructor
	BlellochScan(unsigned int nPlatform); // Basic constructor
	BlellochScan(GpgpuSetup* openCLSetup); // use the BlellochScan with an already existing OpenCL setup
	~BlellochScan();

	//public methods
	void RunBlellochScan(int *& inputData, int *& outputData, const unsigned int & nArraySize);
	void RunBlellochScan(const cl_mem & inputBuffer, cl_mem & outputBuffer, const unsigned int & nArraySize);

	unsigned int GetError();

private:
	//private methods
	void CreateKernels();

	void RecursiveScan(const cl_mem& inputBufferAddress, cl_mem& outputBufferAddress, const unsigned int & nArraySize, cl_mem sumsBuffer, unsigned int&  sumsBufferSize);
	void SimpleScan(const cl_mem& inputBufferAddress, cl_mem& outputBufferAddress, const unsigned int & nArraySize, cl_mem& sumsBuffer, unsigned int& nSumsSize);
	void AddSums(const unsigned int & nArraySize, cl_mem outputBufferAddress, cl_mem sumsBuffer);

	// private members
	GpgpuSetup* m_gpgpuSetup;		// Setup for OpenCl calls
	Kernels* m_kernels;				//container for kernels
};