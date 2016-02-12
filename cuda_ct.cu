#include "cuda_ct.h"

namespace ct {

	namespace cuda {

		cudaPitchedPtr create3dVolumeOnGPU(size_t xSize, size_t ySize, size_t zSize) {
			cudaExtent extent = make_cudaExtent(xSize * sizeof(float), ySize, zSize);
			cudaPitchedPtr ptr;
			cudaMalloc3D(&ptr, extent);
			cudaMemset3D(ptr, 0, extent);
			return ptr;
		}

		void delete3dVolumeOnGPU(cudaPitchedPtr devicePtr) {
			cudaFree(devicePtr.ptr);
		}

		std::shared_ptr<float> download3dVolume(cudaPitchedPtr devicePtr, size_t xSize, size_t ySize, size_t zSize) {
			float* hostDataPtr = new float[xSize * ySize * zSize];
			cudaPitchedPtr hostPtr = make_cudaPitchedPtr(hostDataPtr, xSize * sizeof(float), xSize, ySize);
			cudaExtent extent = make_cudaExtent(xSize * sizeof(float), ySize, zSize);
			cudaMemcpy3DParms memcopyParameters = { 0 };
			memcopyParameters.srcPtr = devicePtr;
			memcopyParameters.dstPtr = hostPtr;
			memcopyParameters.extent = extent;
			memcopyParameters.kind = cudaMemcpyDeviceToHost;
			cudaMemcpy3D(&memcopyParameters);
			return std::shared_ptr<float>(hostDataPtr, std::default_delete<float[]>());
		}

		__global__ void reconstructionKernel(cv::cuda::PtrStepSz<float> const& image, cudaPitchedPtr volumePtr, size_t xSize, size_t ySize, size_t zSize, double radiusSquared, double sine, double cosine, double heightOffset, double uOffset, double SD, double imageLowerBoundU, double imageUpperBoundU, double imageLowerBoundV, double imageUpperBountV, double volumeToWorldXPrecomputed, double volumeToWorldYPrecomputed, double volumeToWorldZPrecomputed, double imageToMatUPrecomputed, double imageToMatVPrecomputed) {
			//do sth
		}

		void startReconstruction(cv::cuda::PtrStepSz<float> const & image, cudaPitchedPtr volumePtr, size_t xSize, size_t ySize, size_t zSize, double radiusSquared, double sine, double cosine, double heightOffset, double uOffset, double SD, double imageLowerBoundU, double imageUpperBoundU, double imageLowerBoundV, double imageUpperBoundV, double volumeToWorldXPrecomputed, double volumeToWorldYPrecomputed, double volumeToWorldZPrecomputed, double imageToMatUPrecomputed, double imageToMatVPrecomputed) {
			dim3 blocks(xSize, ySize, zSize);
			reconstructionKernel <<< blocks, 1 >>>(image,
												   volumePtr,
												   xSize,
												   ySize,
												   zSize,
												   radiusSquared,
												   sine,
												   cosine,
												   heightOffset,
												   uOffset,
												   SD,
												   imageLowerBoundU,
												   imageUpperBoundU,
												   imageLowerBoundV,
												   imageUpperBoundV,
												   volumeToWorldXPrecomputed,
												   volumeToWorldYPrecomputed,
												   volumeToWorldZPrecomputed,
												   imageToMatUPrecomputed,
												   imageToMatVPrecomputed);
		}
		
	}

}