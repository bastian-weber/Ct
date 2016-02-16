#include <cstdlib>
#include <stdio.h>
#include <memory>
#include <cmath>
#include <iostream>

//CUDA
#include <cuda_runtime.h>

//OpenCV
#include <opencv2/core/cuda.hpp>
//#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudev.hpp>

namespace ct {

	namespace cuda {

		size_t getFreeMemory();
		int getMultiprocessorCnt(int deviceId);
		cudaPitchedPtr create3dVolumeOnGPU(size_t xSize, size_t ySize, size_t zSize, bool& success);
		void delete3dVolumeOnGPU(cudaPitchedPtr devicePtr, bool& success);
		std::shared_ptr<float> download3dVolume(cudaPitchedPtr devicePtr, size_t xSize, size_t ySize, size_t zSize, bool& success);
		void startReconstruction(cv::cuda::PtrStepSz<float> image,
								 cudaPitchedPtr volumePtr,
								 size_t xSize,
								 size_t ySize,
								 size_t zSize,
								 size_t zOffset,
								 double radiusSquared,
								 double sine,
								 double cosine,
								 double heightOffset,
								 double uOffset,
								 double SD,
								 double imageLowerBoundU,
								 double imageUpperBoundU,
								 double imageLowerBoundV,
								 double imageUpperBoundV,
								 double volumeToWorldXPrecomputed,
								 double volumeToWorldYPrecomputed,
								 double volumeToWorldZPrecomputed,
								 double imageToMatUPrecomputed,
								 double imageToMatVPrecomputed,
								 bool& success);
		void deviceSynchronize(bool& success);
	
	}

}