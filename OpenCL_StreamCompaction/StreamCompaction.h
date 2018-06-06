#pragma once
class BlellochScan;
class GpgpuSetup;

class StreamCompaction
{
public:
	// I am now trying to remove most dependencies from OpenCL in the header files to make it easier to port it to CUDA if i want to
	struct Kernels;
	enum Predicate {
		EVEN,
		ODD,
		LESSER500
	};

	// constructor destructor
	StreamCompaction(unsigned int nPlatform); // Basic constructor
	StreamCompaction(GpgpuSetup* openCLSetup); // use the StreamCompaction with an already existing OpenCL setup
	~StreamCompaction();
	BlellochScan* m_blellochScan;
	unsigned int GetError();

	// public methods
	int CompactStream(int*& input, int*& output, unsigned int nArraySize, Predicate predicate);

private:

	// private methods
	void CreateKernels();

	// private members

	GpgpuSetup* m_gpgpuSetup;	// Setup for OpenCl calls
	Kernels* m_kernels;			// Holds the kernels
};