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

		__device__ float bilinearInterpolation(double u, double v, float u0v0, float u1v0, float u0v1, float u1v1) {
			//the two interpolations on the u axis
			double v0 = (1.0 - u)*u0v0 + u*u1v0;
			double v1 = (1.0 - u)*u0v1 + u*u1v1;
			//interpolation on the v axis between the two u-interpolated values
			return (1.0 - v)*v0 + v*v1;
		}

		__device__ void addToVolumeElement(cudaPitchedPtr volumePtr, size_t ySize, size_t xCoord, size_t yCoord, size_t zCoord, float value) {
			char* devicePtr = (char*)(volumePtr.ptr);
			//z * xSize * ySize + y * xSize + x
			size_t pitch = volumePtr.pitch;
			size_t slicePitch = pitch * ySize;
			char* slice = devicePtr + zCoord*slicePitch;
			float* row = (float*)(slice + yCoord * pitch);
			row[xCoord] += value;
		}

		__global__ void reconstructionKernel(cv::cuda::PtrStepSz<float> const& image, cudaPitchedPtr volumePtr, size_t xSize, size_t ySize, size_t zSize, double radiusSquared, double sine, double cosine, double heightOffset, double uOffset, double SD, double imageLowerBoundU, double imageUpperBoundU, double imageLowerBoundV, double imageUpperBoundV, double volumeToWorldXPrecomputed, double volumeToWorldYPrecomputed, double volumeToWorldZPrecomputed, double imageToMatUPrecomputed, double imageToMatVPrecomputed) {
			
			size_t xIndex = blockIdx.x;
			size_t yIndex = blockIdx.y;
			size_t zIndex = blockIdx.z;
			
			//just make sure we're inside the volume bounds
			if (xIndex < xSize && yIndex < ySize && zIndex < zSize) {
				//check if voxel is inside the reconstructable cylinder
				if ((xIndex*xIndex + yIndex*yIndex) < radiusSquared) {


				//calculate the world coordinates
					double x = double(xIndex) - volumeToWorldXPrecomputed;
					double y = double(yIndex) - volumeToWorldYPrecomputed;
					double z = double(zIndex) - volumeToWorldZPrecomputed;

					double t = (-1)*x*sine + y*cosine;
					t += uOffset;
					double s = x*cosine + y*sine;
					double u = (t*SD) / (SD - s);
					double v = ((z + heightOffset)*SD) / (SD - s);

					//check if it's inside the image (before the coordinate transformation)
					if (u >= imageLowerBoundU && u <= imageUpperBoundU && v >= imageLowerBoundV && v <= imageUpperBoundV) {

						u += imageToMatUPrecomputed;
						v += imageToMatVPrecomputed;

						//get the 4 surrounding pixels for bilinear interpolation (note: u and v are always positive)
						size_t u0 = u;
						size_t u1 = u0 + 1;
						size_t v0 = v;
						size_t v1 = v0 + 1;

						float u0v0 = image(v0, u0);
						float u1v0 = image(v0, u1);
						float u0v1 = image(v1, u0);
						float u1v1 = image(v1, u1);

						float value = bilinearInterpolation(u - double(u0), v - double(v0), u0v0, u1v0, u0v1, u1v1);
						addToVolumeElement(volumePtr, ySize, xIndex, yIndex, zIndex, value);
					}
				}
			}

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