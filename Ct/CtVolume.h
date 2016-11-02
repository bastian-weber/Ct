#ifndef CT_CTVOLUME
#define CT_CTVOLUME

#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <regex>
#include <fstream>															//used for the output (writing binary file)
#include <cmath>				
#include <ctime>			
#include <future>			
#include <omp.h>

//OpenCV
#include <opencv2/core/core.hpp>											//core functionality of OpenCV
#include <opencv2/highgui/highgui.hpp>										//GUI functionality of OpenCV (display images etc)
#include <opencv2/imgproc/imgproc.hpp>										//OpenCV CUDA
#include <opencv2/cudaarithm.hpp>											

#include "opencv2/core/cuda_stream_accessor.hpp"

//Qt
#include <QtCore/QtCore>

//CUDA
#include <cuda_runtime.h>
#include <cufft.h>

#include "Volume.h"
#include "utility.h"
#include "types.h"
#include "cuda_ct.h"

//undefines some macros of VC++ that cause naming conflicts
//for std::numeric_limits<std::streamsize>::max()
#undef max
#undef min

namespace ct {

	//struct for returning projections
	struct Projection {
		Projection();
		Projection(cv::Mat image, double angle, double heightOffset);
		cv::Mat image;
		double angle;
		double heightOffset;													//for random trajectory
	};

	enum class FilterType {
		RAMLAK,
		SHEPP_LOGAN,
		HANN
	};

	//The actual reconstruction class
	class CtVolume : public QObject {
		Q_OBJECT
	public:

		//=========================================== PUBLIC FUNCTIONS ===========================================\\

		//constructors
		CtVolume();
		CtVolume(QString csvFile);
		//getters
		static bool cudaAvailable(bool verbose = false);
		static int getCudaDeviceCount(bool verbose = false);
		ct::Projection getProjectionAt(size_t index) const;
		size_t getSinogramSize() const;
		unsigned int getImageWidth() const;
		unsigned int getImageHeight() const;
		unsigned int getXSize() const;
		unsigned int getYSize() const;
		unsigned int getZSize() const;
		unsigned int getReconstructionCylinderRadius() const;
		float getUOffset() const;
		float getPixelSize() const;
		float getFCD() const;
		float getBaseIntensity() const;
		cv::Mat getVolumeCrossSection(Axis axis, unsigned int index) const;
		unsigned int getCrossSectionIndex() const;
		unsigned int getCrossSectionSize() const;
		Axis getCrossSectionAxis() const;
		bool getEmitSignals() const;
		bool getUseCuda() const;
		unsigned int getSkipProjections() const;
		std::vector<int> getActiveCudaDevices() const;
		std::vector<std::string> getCudaDeviceList() const;
		size_t getGpuSpareMemory() const;
		double getMultiprocessorCoefficient() const;
		double getMemoryBandwidthCoefficient() const;
		size_t getRequiredMemoryUpperBound() const;
		bool getUseCpuPreprocessing() const;

		//setters
		void setCrossSectionIndex(unsigned int index);
		void setCrossSectionAxis(Axis axis);
		void setEmitSignals(bool value);
		void setVolumeBounds(float xFrom,
							 float xTo,
							 float yFrom,
							 float yTo,
							 float zFrom,
							 float zTo);
		void setUseCuda(bool value);
		void setActiveCudaDevices(std::vector<int> devices);
		void setGpuSpareMemory(size_t memory);									//sets the amount of VRAM to spare in Mb
		void setGpuCoefficients(double multiprocessorCoefficient,
								double memoryBandwidthCoefficient);
		void setUseCpuPreprocessing(bool value);
		void setFrequencyFilterType(FilterType filterType);
		void setSkipProjections(unsigned int value = 0);

		//control functions
		bool sinogramFromImages(QString csvFile);								//creates a sinogramm out of images specified in csvFile								
		void reconstructVolume();												//reconstructs the 3d-volume from the sinogram, filterType specifies the used prefilter
		void saveVolumeToBinaryFile(QString filename,							//saves the reconstructed volume to a binary file
									IndexOrder indexOrder = IndexOrder::Z_FASTEST,
									QDataStream::ByteOrder byteOrder = QDataStream::LittleEndian) const;					
		void stop();															//should stop the operation that's currently running (either preprocessing, reconstruction or saving)

	private:

		//struct for storing one projection internally
		struct Projection {
			Projection();
			Projection(QString imagePath, double angle, double heightOffset);	//Constructor
			cv::Mat getImage() const;
			ct::Projection getPublicProjection() const;
			QString imagePath;
			double angle;
			double heightOffset;													//for random trajectory
		};

		//FFT filter class
		class FftFilter {
		public:
			FftFilter() = delete;
			FftFilter(FftFilter const& other);
			FftFilter(int width, int height);
			~FftFilter();
			FftFilter& operator=(FftFilter const& other) = delete;
			bool good();
			static size_t getWorkSizeEstimate(int width, int height);
			void setStream(cudaStream_t stream, bool& success);
			void applyForward(cv::cuda::GpuMat& imageIn, cv::cuda::GpuMat& dftSpectrumOut, bool& success) const;
			void applyInverse(cv::cuda::GpuMat& dftSpectrumIn, cv::cuda::GpuMat& imageOut, bool& success) const;
		private:
			cufftHandle forwardPlan;
			cufftHandle inversePlan;
			bool isGood = true;
			int width;
			int height;
		};
		
		//=========================================== PRIVATE FUNCTIONS ===========================================\\
		
		//related to parsing of config file
		void initialise();
		void makeHeightOffsetRelative();
		void correctAngleDirection(std::string rotationDirection);
		
		//related to CPU image preprocessing
		cv::Mat normalizeImage(cv::Mat const& image,						//returns a new image which is a version of the old image that is normalized by min and max value
							   float minValue,
							   float maxValue) const;
		cv::Mat prepareProjection(unsigned int index,						//returns the image of the projection at position index preprocessed and converted
								  bool multithreading = false) const;						
		void preprocessImage(cv::Mat& image,
							 bool multithreading = false) const;
		void convertTo32bit(cv::Mat& img) const;							//converts an image to 32bit float
		void applyFeldkampWeight(cv::Mat& image,
								 bool multithreading = false) const;
		static float W(float D, float u, float v);							//weight function for the reconstruction of the volume		
		static void applyFourierFilter(cv::Mat& image, 
									   FilterType filterType,
									   bool multithreading = false);
		void applyLogScaling(cv::Mat& image) const;								//applies a logarithmic scaling to an image
		static float ramLakWindowFilter(float n, float N);					//these functions return the scaling coefficients for the
		static float sheppLoganWindowFilter(float n, float N);				//fourier filters for each n out of N
		static float hannWindowFilter(float n, float N);
		
		//related to the GPU image preprocessing
		void cudaPreprocessImage(cv::cuda::GpuMat& imageIn,
								 cv::cuda::GpuMat& imageOut,
								 cv::cuda::GpuMat& dftTmp,
								 FftFilter& fftFilter,
								 bool& success,
								 cv::cuda::Stream& stream = cv::cuda::Stream::Null()) const;

		//related to the CPU reconstruction
		bool reconstructionCore();											//does the actual reconstruction
		static float bilinearInterpolation(float u,							//interpolates bilinear between those four intensities
										   float v,
										   float u0v0,
										   float u1v0,
										   float u0v1,
										   float u1v1);

		//related to the GPU reconstruction
		bool cudaReconstructionCore(unsigned int threadZMin,
									unsigned int threadZMax,
									int deviceId);
		bool launchCudaThreads();
		std::map<int, double> getGpuWeights(std::vector<int> const& devices) const;
		unsigned int getMaxChunkSize() const;							//returns the maximum amount of slices in z-direction that fit into VRAM for current GPU

		//coordinate transformation functions
		void updateBoundaries();												//is called when the bounds of the ROI change, precomputes some values
		float worldToVolumeX(float xCoord) const;								//coordinate transformations from the coordinates of the vector to
		float worldToVolumeY(float yCoord) const;								//the coordinates of the "world" and the other way around
		float worldToVolumeZ(float zCoord) const;
		float volumeToWorldX(float xCoord) const;
		float volumeToWorldY(float yCoord) const;
		float volumeToWorldZ(float zCoord) const;
		float imageToMatU(float uCoord)const;									//coordinate transformations from the coordinates of the image
		float imageToMatV(float vCoord)const;									//to the coordinates of the saved matrix (always starting at 0)
		float matToImageU(float uCoord)const;
		float matToImageV(float vCoord)const;
		
		//=========================================== PRIVATE VARIABLES ===========================================\\

		//parameters that are set when reading the config file/the images
		std::vector<Projection> sinogram;									//here the images are stored
		unsigned int xSize = 0, ySize = 0, zSize = 0;						//the size of the volume in x, y and z direction, is calculated when sinogram is created
		unsigned int imageWidth = 0, imageHeight = 0;						//stores the height and width of the images in the sinogram
		int imageType;														//assumed type of all the images (taken from the first image)
		float FCD = 0;														//the distance of the source to the detector in pixel
		float pixelSize = 0;
		float uOffset = 0;													//the offset of the rotation axis in u direction																			//bounds of what will be reconstructed
		float baseIntensity = 0;											//the intensity of just air

		//variables that can be set from outside and controls the behaviour of the object
		FilterType filterType = FilterType::RAMLAK;							//holds the frequency filter type that shall be used
		unsigned int projectionStep = 1;									//controls how many projections are skipped during reconstruction
		bool emitSignals = true;											//if true the object emits qt signals in certain functions
		unsigned int crossSectionIndex = 0;									//index for the crossection that is returned in qt signals
		Axis crossSectionAxis = Axis::Z;									//the axis at which the volume is sliced for the cross section
		bool useCuda = true;												//enables or disables the use of cuda
		std::vector<int> activeCudaDevices;									//containing the deviceIds of the gpus that shall be used
		size_t gpuSpareMemory = 200;										//the amount of gpu memory to spare in Mb
		double multiprocessorCoefficient = 1, memoryBandwidthCoefficient = 1;//conrols the weighting of multiprocessor count and memory speed amongst multiple gpus
		bool useCpuPreprocessing = false;
		mutable std::atomic<bool> stopActiveProcess{ false };				//is set to true when stop() is called
		float xFrom_float = 0, xTo_float = 1;								//these values control the ROI of the volume that is reconstructed within 0 <= x <= 1
		float yFrom_float = 0, yTo_float = 1;
		float zFrom_float = 0, zTo_float = 1;

		//variables that are only internally used
		Volume<float> volume;												//holds the reconstructed volume
		unsigned int xFrom = 0, xTo = 0;									//the volume ROI in actual volume coordinates, calculated from the float ROI
		unsigned int yFrom = 0, yTo = 0;
		unsigned int zFrom = 0, zTo = 0;
		unsigned int xMax = 0, yMax = 0, zMax = 0;							//the width, height and depth of the volume ROI that is going to be reconstructed
		mutable std::pair<float, float> minMaxValues;						//stores the brightest and darkes value in all of the sinogram images (for normalisation)
		std::map<int, double> cudaThreadProgress, cudaGpuWeights;			//for keeping track of the progress on multithread CUDA execution
		mutable std::mutex exclusiveFunctionsMutex;							//this mutex makes sure certain functions are not executed concurrently
		mutable std::atomic<bool> stopCudaThreads{ false };					//this will be set to true if all cuda threads shall be stopped
		std::string lastErrorMessage;										//if an error in one of the threads occurs, it will be saved here

		//variables for precomputed parts of coordinate transformations
		float xPrecomputed;
		float yPrecomputed;
		float zPrecomputed;
		float uPrecomputed;
		float vPrecomputed;
	private slots:
		void emitGlobalCudaProgress(double percentage, int deviceId, bool emitCrossSection);
	signals:
		void loadingProgress(double percentage) const;
		void loadingFinished(CompletionStatus status = CompletionStatus::success()) const;
		void reconstructionProgress(double percentage, cv::Mat crossSection) const;
		void reconstructionFinished(cv::Mat crossSection, CompletionStatus status = CompletionStatus::success()) const;
		void savingProgress(double percentage) const;
		void savingFinished(CompletionStatus status = CompletionStatus::success()) const;
		void cudaThreadProgressUpdate(double percentage, int deviceId, bool emitCrossSection);
	};

}

#endif