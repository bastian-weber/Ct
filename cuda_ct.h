#include <cstdlib>
#include <stdio.h>
#include <memory>
#include <cmath>

//CUDA
#include <cuda_runtime.h>

//OpenCV
#include <opencv2/core/cuda.hpp>
//#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudev.hpp>

namespace ct {

	namespace cuda {

		cudaPitchedPtr create3dVolumeOnGPU(size_t xSize, size_t ySize, size_t zSize);
		void delete3dVolumeOnGPU(cudaPitchedPtr devicePtr);
		std::shared_ptr<float> download3dVolume(cudaPitchedPtr devicePtr, size_t xSize, size_t ySize, size_t zSize);
		void startReconstruction(cv::cuda::PtrStepSz<float> image,
								 cudaPitchedPtr volumePtr,
								 size_t xSize,
								 size_t ySize,
								 size_t zSize,
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
								 double imageToMatVPrecomputed);
		void deviceSynchronize();
	
	}

}