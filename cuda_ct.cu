#include "cuda_ct.h"

namespace ct {

	namespace cuda {

		int getMultiprocessorCnt(int deviceId) {
			int cnt;
			cudaDeviceGetAttribute(&cnt, cudaDevAttrMultiProcessorCount, deviceId);
			return cnt;
		}
		cudaPitchedPtr create3dVolumeOnGPU(size_t xSize, size_t ySize, size_t zSize, bool& success) {
			success = true;
			cudaError_t status;
			cudaExtent extent = make_cudaExtent(xSize * sizeof(float), ySize, zSize);
			cudaPitchedPtr ptr;
			status = cudaMalloc3D(&ptr, extent);
			if (status != cudaSuccess) {
				std::cout << "cudaMalloc3D ERROR: " << cudaGetErrorString(status) << std::endl;
				success = false;
			}
			status = cudaMemset3D(ptr, 0, extent);
			if (status != cudaSuccess) {
				std::cout << "cudaMemset3D ERROR: " << cudaGetErrorString(status) << std::endl;
				success = false;
			}
			return ptr;
		}

		void delete3dVolumeOnGPU(cudaPitchedPtr devicePtr, bool& success) {
			success = true;
			cudaError_t status = cudaFree(devicePtr.ptr);
			if (status != cudaSuccess) {
				std::cout << "cudaFree ERROR: " << cudaGetErrorString(status) << std::endl;
				success = false;
			}
		}

		std::shared_ptr<float> download3dVolume(cudaPitchedPtr devicePtr, size_t xSize, size_t ySize, size_t zSize, bool& success) {
			success = true;
			float* hostDataPtr = new float[xSize * ySize * zSize];
			cudaPitchedPtr hostPtr = make_cudaPitchedPtr(hostDataPtr, xSize * sizeof(float), xSize, ySize);
			cudaExtent extent = make_cudaExtent(xSize * sizeof(float), ySize, zSize);
			cudaMemcpy3DParms memcopyParameters = { 0 };
			memcopyParameters.srcPtr = devicePtr;
			memcopyParameters.dstPtr = hostPtr;
			memcopyParameters.extent = extent;
			memcopyParameters.kind = cudaMemcpyDeviceToHost;
			cudaError_t status = cudaMemcpy3D(&memcopyParameters);
			if (status != cudaSuccess) {
				std::cout << "cudaMemcpy3D ERROR: " << cudaGetErrorString(status) << std::endl;
				success = false;
			}
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

		__global__ void reconstructionKernel(cv::cuda::PtrStepSz<float> image, cv::cuda::PtrStepSz<float> lookupTable, cudaPitchedPtr volumePtr, size_t xSize, size_t ySize, size_t zSize, size_t zOffset, double radiusSquared, double sine, double cosine, double heightOffset, double uOffset, double SD, double imageLowerBoundU, double imageUpperBoundU, double imageLowerBoundV, double imageUpperBoundV, double imageToMatUPrecomputed, double imageToMatVPrecomputed) {

			size_t xIndex = threadIdx.x + blockIdx.x * blockDim.x;
			size_t yIndex = threadIdx.y + blockIdx.y * blockDim.y;
			size_t zIndex = threadIdx.z + blockIdx.z * blockDim.z;

			//if (xIndex == 0 && yIndex == 0 && zIndex == 0) {
			//	printf("kernel start\n");
			//}

			//just make sure we're inside the volume bounds
			if (xIndex < xSize && yIndex < ySize && zIndex < zSize) {

				//calculate the world coordinates
				double x = lookupTable(0, xIndex);
				double y = lookupTable(1, yIndex);
				double z = lookupTable(2, zIndex + zOffset);

				//check if voxel is inside the reconstructable cylinder
				if ((x*x + y*y) < radiusSquared) {

					double t = (-1)*x*sine + y*cosine;
					t += uOffset;
					double s = x*cosine + y*sine;
					double u = (t*SD) / (SD - s);
					double v = ((z + heightOffset)*SD) / (SD - s);

					//check if it's inside the image (before the coordinate transformation)
					if (u >= imageLowerBoundU && u <= imageUpperBoundU && v >= imageLowerBoundV && v <= imageUpperBoundV) {

						u += imageToMatUPrecomputed;
						v = (-1)*v + imageToMatVPrecomputed;

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

			//if (xIndex == 0 && yIndex == 0 && zIndex == 0) {
			//	printf("kernel end\n");
			//}

		}

		void startReconstruction(cv::cuda::PtrStepSz<float> image, 
								 cv::cuda::PtrStepSz<float> lookupTable,
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
								 double imageToMatUPrecomputed, 
								 double imageToMatVPrecomputed, 
								 bool& success) {
			success = true;
			dim3 threads(32, 32, 1);
			dim3 blocks(((unsigned int)xSize + threads.x - 1) / threads.x,
						((unsigned int)ySize + threads.y - 1) / threads.y,
						((unsigned int)zSize + threads.z - 1) / threads.z);
			reconstructionKernel << < blocks, threads >> >(image,
														   lookupTable,
														   volumePtr,
														   xSize,
														   ySize,
														   zSize,
														   zOffset,
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
														   imageToMatUPrecomputed,
														   imageToMatVPrecomputed);
			cudaError_t status = cudaGetLastError();
			if (status != cudaSuccess) {
				std::cout << std::endl << "Kernel launch ERROR: " << cudaGetErrorString(status);
				success = false;
			}
		}

		void deviceSynchronize(bool& success) {
			success = true;
			cudaError_t status = cudaDeviceSynchronize();
			if (status != cudaSuccess) {
				std::cout << std::endl << "cudaDeviceSynchronize ERROR: " << cudaGetErrorString(status);
				success = false;
			}
		}

	}

}