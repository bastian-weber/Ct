#include "cuda_ct.h"
#include "math_constants.h"

namespace ct {

	namespace cuda {

		size_t getFreeMemory() {
			size_t freeMemory, totalMemory;
			cudaMemGetInfo(&freeMemory, &totalMemory);
			return freeMemory;
		}

		size_t getTotalMemory() {
			size_t freeMemory, totalMemory;
			cudaMemGetInfo(&freeMemory, &totalMemory);
			return totalMemory;
		}

		int getMultiprocessorCnt(int deviceId) {
			int cnt;
			cudaDeviceGetAttribute(&cnt, cudaDevAttrMultiProcessorCount, deviceId);
			return cnt;
		}

		int getMemoryBusWidth(int deviceId) {
			int busWidth;
			cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, deviceId);
			return busWidth;
		}

		int getMemoryClockRate(int deviceId) {
			int clockRate;
			cudaDeviceGetAttribute(&clockRate, cudaDevAttrMemoryClockRate, deviceId);
			return clockRate;
		}

		std::string getDeviceName(int deviceId) {
			cudaSetDevice(deviceId);
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, deviceId);
			double memory = getTotalMemory();
			std::ostringstream out;
			out << std::setprecision(1) << std::fixed << prop.name << ", " << memory / 1024.0 / 1024.0 / 1024.0 << " Gb";
			return out.str();
		}

		__device__ float ramLakWindowFilter(float n, float Nreciprocal){
			return n * Nreciprocal;
		}

		__device__ float sheppLoganWindowFilter(float n, float Nreciprocal) {
			if (n == 0.0f) {
				return 0.0f;
			} else {
				float rl = ramLakWindowFilter(n, Nreciprocal);
				return (rl)* (__sinf(rl*0.5f*CUDART_PI_F)) / (rl*0.5f*CUDART_PI_F);
			}

		}

		__device__ float hannWindowFilter(float n, float Nreciprocal) {
			float rl = ramLakWindowFilter(n, Nreciprocal);
			return rl * 0.5f*(1.0f + __cosf(CUDART_PI_F * rl));
		}

		__global__ void frequencyFilterKernel(cv::cuda::PtrStepSz<float2> image, int filterType) {

			size_t xIndex = threadIdx.x + blockIdx.x * blockDim.x;
			size_t yIndex = threadIdx.y + blockIdx.y * blockDim.y;

			if (xIndex < image.cols && yIndex < image.rows) {
				float2 pixel = image(yIndex, xIndex);
				float Nreciprocal = 1.0 / static_cast<float>(image.cols);
				float factor;
				if (filterType == 0) {
					factor = ramLakWindowFilter(xIndex, Nreciprocal);
				} else if (filterType == 1) {
					factor = sheppLoganWindowFilter(xIndex, Nreciprocal);
				} else if (filterType == 2) {
					factor = hannWindowFilter(xIndex, Nreciprocal);
				}
				image(yIndex, xIndex) = make_float2(pixel.x*factor, pixel.y*factor);
			}

		}

		void applyFrequencyFiltering(cv::cuda::PtrStepSz<float2> image, int filterType, cudaStream_t stream, bool& success) {
			success = true;
			dim3 threads(32, 1);
			dim3 blocks(std::ceil(float(image.cols) / float(threads.x)),
						std::ceil(float(image.rows) / float(threads.y)));
			frequencyFilterKernel << < blocks, threads, 0, stream >> >(image, filterType);

			cudaError_t status = cudaGetLastError();
			if (status != cudaSuccess) {
				std::cout << std::endl << "Kernel launch ERROR: " << cudaGetErrorString(status);
				success = false;
			}
		}

		__device__ float W(float D, float u, float v) {
			return D * rsqrtf(D*D + u*u + v*v);
		}

		__global__ void feldkampWeightFilterKernel(cv::cuda::PtrStepSz<float> image, float SD, float uPrecomputed, float vPrecomputed) {

			size_t xIndex = threadIdx.x + blockIdx.x * blockDim.x;
			size_t yIndex = threadIdx.y + blockIdx.y * blockDim.y;

			if (xIndex < image.cols && yIndex < image.rows) {
				float u = float(xIndex) - uPrecomputed;
				float v = -float(yIndex) + vPrecomputed;
				image(yIndex, xIndex) *= W(SD, u, v);
			}

		}

		void applyFeldkampWeightFiltering(cv::cuda::PtrStepSz<float> image, float SD, float uPrecomputed, float vPrecomputed, cudaStream_t stream, bool& success) {
			success = true;
			dim3 threads(32, 1);
			dim3 blocks(std::ceil(float(image.cols) / float(threads.x)),
						std::ceil(float(image.rows) / float(threads.y)));
			feldkampWeightFilterKernel << < blocks, threads, 0, stream >> >(image, SD, uPrecomputed, vPrecomputed);

			cudaError_t status = cudaGetLastError();
			if (status != cudaSuccess) {
				std::cout << std::endl << "Kernel launch ERROR: " << cudaGetErrorString(status);
				success = false;
			}
		}

		cudaPitchedPtr create3dVolumeOnGPU(size_t xSize, size_t ySize, size_t zSize, bool& success, bool verbose) {
			success = true;
			cudaError_t status;
			cudaExtent extent = make_cudaExtent(xSize * sizeof(float), ySize, zSize);
			cudaPitchedPtr ptr;
			status = cudaMalloc3D(&ptr, extent);
			if (status != cudaSuccess) {
				if (verbose) std::cout << "cudaMalloc3D ERROR: " << cudaGetErrorString(status) << std::endl;
				success = false;
			}
			//if something went wrong try to deallocate memory
			if (!success) cudaFree(ptr.ptr);
			return ptr;
		}

		void setToZero(cudaPitchedPtr devicePtr, size_t xSize, size_t ySize, size_t zSize, bool& success) {
			success = true;
			cudaError_t status;
			cudaExtent extent = make_cudaExtent(xSize * sizeof(float), ySize, zSize);		
			status = cudaMemset3D(devicePtr, 0, extent);
			if (status != cudaSuccess) {
				std::cout << "cudaMemset3D ERROR: " << cudaGetErrorString(status) << std::endl;
				success = false;
			}
		}

		void delete3dVolumeOnGPU(cudaPitchedPtr devicePtr, bool& success) {
			success = true;
			cudaError_t status = cudaFree(devicePtr.ptr);
			if (status != cudaSuccess) {
				std::cout << "cudaFree ERROR: " << cudaGetErrorString(status) << std::endl;
				success = false;
			}
		}

		void download3dVolume(cudaPitchedPtr devicePtr, float* hostPtr, size_t xSize, size_t ySize, size_t zSize,  bool& success) {
			success = true;
			cudaPitchedPtr hostPitchedPtr = make_cudaPitchedPtr(hostPtr, xSize * sizeof(float), xSize, ySize);
			cudaExtent extent = make_cudaExtent(xSize * sizeof(float), ySize, zSize);
			cudaMemcpy3DParms memcopyParameters = { 0 };
			memcopyParameters.srcPtr = devicePtr;
			memcopyParameters.dstPtr = hostPitchedPtr;
			memcopyParameters.extent = extent;
			memcopyParameters.kind = cudaMemcpyDeviceToHost;
			cudaError_t status = cudaMemcpy3D(&memcopyParameters);
			if (status != cudaSuccess) {
				std::cout << "cudaMemcpy3D ERROR: " << cudaGetErrorString(status) << std::endl;
				success = false;
			}
		}

		__device__ float bilinearInterpolation(float u, float v, float u0v0, float u1v0, float u0v1, float u1v1) {
			//the two interpolations on the u axis
			float v0 = (1.0f - u)*u0v0 + u*u1v0;
			float v1 = (1.0f - u)*u0v1 + u*u1v1;
			//interpolation on the v axis between the two u-interpolated values
			return (1.0f - v)*v0 + v*v1;
		}

		__device__ void addToVolumeElement(cudaPitchedPtr volumePtr, size_t xCoord, size_t yCoord, size_t zCoord, float value) {
			char* devicePtr = (char*)(volumePtr.ptr);
			//z * xSize * ySize + y * xSize + x
			size_t pitch = volumePtr.pitch;
			size_t slicePitch = pitch * volumePtr.ysize;
			char* slice = devicePtr + zCoord*slicePitch;
			float* row = (float*)(slice + yCoord * pitch);
			row[xCoord] += value;
		}

		__global__ void reconstructionKernel(cv::cuda::PtrStepSz<float> image, 
											 cudaPitchedPtr volumePtr, 
											 size_t xSize, size_t ySize, 
											 size_t zSize, size_t zOffset, 
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
											 float xPrecomputed,
											 float yPrecomputed,
											 float zPrecomputed,
											 float uPrecomputed,
											 float vPrecomputed) {

			size_t xIndex = threadIdx.x + blockIdx.x * blockDim.x;
			size_t yIndex = threadIdx.y + blockIdx.y * blockDim.y;
			size_t zIndex = threadIdx.z + blockIdx.z * blockDim.z;

			//if (xIndex == 0 && yIndex == 0 && zIndex == 0) {
			//	printf("kernel start\n");
			//}

			//just make sure we're inside the volume bounds
			if (xIndex < xSize && yIndex < ySize && zIndex < zSize) {

				//calculate the world coordinates
				float x = float(xIndex) - xPrecomputed;
				float y = float(yIndex) - yPrecomputed;
				float z = float(zIndex + zOffset) - zPrecomputed;

				//check if voxel is inside the reconstructable cylinder
				if ((x*x + y*y) < radiusSquared) {

					float t = -x*sine + y*cosine;
					t += uOffset;
					float s = x*cosine + y*sine;
					float u = (t*SD) / (SD - s);
					float v = ((z + heightOffset)*SD) / (SD - s);

					//check if it's inside the image (before the coordinate transformation)
					if (u >= imageLowerBoundU && u <= imageUpperBoundU && v >= imageLowerBoundV && v <= imageUpperBoundV) {

						//calculate weight
						float w = SD / (SD + s);
						w = w*w;

						u += uPrecomputed;
						v = -v + vPrecomputed;

						//get the 4 surrounding pixels for bilinear interpolation (note: u and v are always positive)
						size_t u0 = u;
						size_t u1 = u0 + 1;
						size_t v0 = v;
						size_t v1 = v0 + 1;

						float* row = image.ptr(v0);
						float u0v0 = row[u0];
						float u1v0 = row[u1];
						row = image.ptr(v1);
						float u0v1 = row[u0];
						float u1v1 = row[u1];

						float value = w * bilinearInterpolation(u - float(u0), v - float(v0), u0v0, u1v0, u0v1, u1v1);

						addToVolumeElement(volumePtr, xIndex, yIndex, zIndex, value);
					}
				}
			}

			//if (xIndex == 0 && yIndex == 0 && zIndex == 0) {
			//	printf("kernel end\n");
			//}

		}

		void startReconstruction(cv::cuda::PtrStepSz<float> image, 
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
								 float xPrecomputed,
								 float yPrecomputed,
								 float zPrecomputed,
								 float uPrecomputed,
								 float vPrecomputed,
								 cudaStream_t stream,
								 bool& success) {
			success = true;
			dim3 threads(16, 16, 1);
			dim3 blocks(std::ceil(float(xSize) / float(threads.x)),
						std::ceil(float(ySize) / float(threads.y)),
						std::ceil(float(zSize) / float(threads.z)));
			reconstructionKernel << < blocks, threads, 0, stream >> >(image,
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
																	  xPrecomputed,
																	  yPrecomputed,
																	  zPrecomputed,
																	  uPrecomputed,
																	  vPrecomputed);
			cudaError_t status = cudaGetLastError();
			if (status != cudaSuccess) {
				std::cout << std::endl << "Kernel launch ERROR: " << cudaGetErrorString(status);
				success = false;
			}
		}

	}

}