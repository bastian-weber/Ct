#include <cstdlib>
#include <stdio.h>
#include <memory>
#include <cmath>
#include <iostream>
#include <iomanip>

//CUDA
#include <cuda_runtime.h>

//OpenCV
#include <opencv2/core/cuda.hpp>

namespace ct {

	namespace cuda {

		size_t getFreeMemory();														//returns the free memory on the currently active device
		size_t getTotalMemory();													//returns the global memory of the currently active device
		int getMultiprocessorCnt(int deviceId);										//returns the amount of streaming processors
		int getMemoryBusWidth(int deviceId);										//returns the global memory bus width
		int getMemoryClockRate(int deviceId);
		std::string getDeviceName(int deviceId);									//returns the name of the device
		void applyFrequencyFiltering(cv::cuda::PtrStepSz<float2> image,				//applies a frequency filter on a dft spectrum
									 int filterType, 
									 cudaStream_t stream, 
									 bool& success);
		void applyFeldkampWeightFiltering(cv::cuda::PtrStepSz<float> image,			//applies the feldkamp weighting to an image
										  float FCD, 
										  float uPrecomputed, 
										  float vPrecomputed, 
										  cudaStream_t stream, 
										  bool& success);
		cudaPitchedPtr create3dVolumeOnGPU(unsigned int xSize, unsigned int ySize,				//allocates a 3d volume on the gpu and initialises it with 0
										   unsigned int zSize,
										   bool& success,
										   bool verbose = true);
		void setToZero(cudaPitchedPtr devicePtr, unsigned int xSize, unsigned int ySize,		//sets the volume to 0
					   unsigned int zSize, bool& success);
		void delete3dVolumeOnGPU(cudaPitchedPtr devicePtr, bool& success);			//deletes the 3d volume from the gpu memory
		void download3dVolume(cudaPitchedPtr devicePtr,
							  float* hostPtr,
							  unsigned int xSize,
							  unsigned int ySize,
							  unsigned int zSize,
							  bool& success);
		void startReconstruction(cv::cuda::PtrStepSz<float> image,					//starts the reconstruction
								 cudaPitchedPtr volumePtr,
								 unsigned int xSize,
								 unsigned int ySize,
								 unsigned int zSize,
								 unsigned int zOffset,
								 float radiusSquared,
								 float sine,
								 float cosine,
								 float heightOffset,
								 float uOffset,
								 float FCD,
								 float imageLowerBoundU,
								 float imageUpperBoundU,
								 float imageLowerBoundV,
								 float imageUpperBoundV,
								 float xPrecomputed,
								 float yPrecomputed,
								 float zPrecomputed,
								 float uPrecomputed,
								 float vPrecomputed,
								 cudaStream_t stream,
								 bool& success);
	
	}

}