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
										  float SD, 
										  float matToImageUPreprocessed, 
										  float matToImageVPreprocessed, 
										  cudaStream_t stream, 
										  bool& success);
		cudaPitchedPtr create3dVolumeOnGPU(size_t xSize, size_t ySize,				//allocates a 3d volume on the gpu and initialises it with 0
										   size_t zSize, 
										   bool& success);
		void delete3dVolumeOnGPU(cudaPitchedPtr devicePtr, bool& success);			//deletes the 3d volume from the gpu memory
		void download3dVolume(cudaPitchedPtr devicePtr,
							  float* hostPtr,
							  size_t xSize,
							  size_t ySize,
							  size_t zSize,
							  bool& success);
		void startReconstruction(cv::cuda::PtrStepSz<float> image,					//starts the reconstruction
								 cudaPitchedPtr volumePtr,
								 size_t xSize,
								 size_t ySize,
								 size_t zSize,
								 size_t zOffset,
								 float radiusSquared,
								 float sine,
								 float cosine,
								 float heightOffset,
								 float uOffset,
								 float SD,
								 float imageLowerBoundU,
								 float imageUpperBoundU,
								 float imageLowerBoundV,
								 float imageUpperBoundV,
								 float volumeToWorldXPrecomputed,
								 float volumeToWorldYPrecomputed,
								 float volumeToWorldZPrecomputed,
								 float imageToMatUPrecomputed,
								 float imageToMatVPrecomputed,
								 cudaStream_t stream,
								 bool& success);
	
	}

}