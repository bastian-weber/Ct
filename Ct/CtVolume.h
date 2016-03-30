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
#include "CompletionStatus.h"
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
		size_t getImageWidth() const;
		size_t getImageHeight() const;
		size_t getXSize() const;
		size_t getYSize() const;
		size_t getZSize() const;
		double getUOffset() const;
		double getVOffset() const;
		double getPixelSize() const;
		double getSO() const;
		double getSD() const;
		cv::Mat getVolumeCrossSection(Axis axis, size_t index) const;
		size_t getCrossSectionIndex() const;
		size_t getCrossSectionSize() const;
		Axis getCrossSectionAxis() const;
		bool getEmitSignals() const;
		bool getUseCuda() const;
		std::vector<int> getActiveCudaDevices() const;
		std::vector<std::string> getCudaDeviceList() const;
		size_t getGpuSpareMemory() const;
		size_t getRequiredMemoryUpperBound() const;

		//setters
		void setCrossSectionIndex(size_t index);
		void setCrossSectionAxis(Axis axis);
		void setEmitSignals(bool value);
		void setVolumeBounds(double xFrom,
							 double xTo,
							 double yFrom,
							 double yTo,
							 double zFrom,
							 double zTo);
		void setUseCuda(bool value);
		void setActiveCudaDevices(std::vector<int> devices);
		void setGpuSpareMemory(size_t memory);									//sets the amount of VRAM to spare in Mb
		void setFrequencyFilterType(FilterType filterType);

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
		cv::Mat normalizeImage(cv::Mat const& image,							//returns a new image which is a version of the old image that is normalized by min and max value
							   float minValue,
							   float maxValue) const;
		cv::Mat prepareProjection(size_t index) const;							//returns the image of the projection at position index preprocessed and converted
		void preprocessImage(cv::Mat& image) const;
		static void convertTo32bit(cv::Mat& img);								//converts an image to 32bit float
		void applyFeldkampWeight(cv::Mat& image) const;
		static double W(double D, double u, double v);							//weight function for the reconstruction of the volume		
		static void applyFourierFilter(cv::Mat& image, FilterType filterType);
		static void applyLogScaling(cv::Mat& image);							//applies a logarithmic scaling to an image
		static double ramLakWindowFilter(double n, double N);					//these functions return the scaling coefficients for the
		static double sheppLoganWindowFilter(double n, double N);				//fourier filters for each n out of N
		static double hannWindowFilter(double n, double N);						
		
		//related to the GPU image preprocessing
		void cudaPreprocessImage(cv::cuda::GpuMat& imageIn,
								 cv::cuda::GpuMat& imageOut,
								 cv::cuda::GpuMat& dftTmp,
								 FftFilter& fftFilter,
								 bool& success,
								 cv::cuda::Stream& stream = cv::cuda::Stream::Null()) const;

		//related to the CPU reconstruction
		bool reconstructionCore();												//does the actual reconstruction
		static float bilinearInterpolation(double u,							//interpolates bilinear between those four intensities
										   double v,
										   float u0v0,
										   float u1v0,
										   float u0v1,
										   float u1v1);

		//related to the GPU reconstruction
		bool cudaReconstructionCore(size_t threadZMin, 
									size_t threadZMax, 
									int deviceId);
		bool launchCudaThreads();
		std::map<int, double> getGpuWeights(std::vector<int> const& devices) const;
		size_t getMaxChunkSize() const;											//returns the maximum amount of slices in z-direction that fit into VRAM for current GPU

		//coordinate transformation functions
		void updateBoundaries();												//is called when the bounds of the ROI change, precomputes some values
		double worldToVolumeX(double xCoord) const;								//coordinate transformations from the coordinates of the vector to
		double worldToVolumeY(double yCoord) const;								//the coordinates of the "world" and the other way around
		double worldToVolumeZ(double zCoord) const;
		double volumeToWorldX(double xCoord) const;
		double volumeToWorldY(double yCoord) const;
		double volumeToWorldZ(double zCoord) const;
		double imageToMatU(double uCoord)const;									//coordinate transformations from the coordinates of the image
		double imageToMatV(double vCoord)const;									//to the coordinates of the saved matrix (always starting at 0)
		double matToImageU(double uCoord)const;
		double matToImageV(double vCoord)const;
		
		//=========================================== PRIVATE VARIABLES ===========================================\\

		//parameters that are set when reading the config file/the images
		std::vector<Projection> sinogram;									//here the images are stored
		size_t xSize = 0, ySize = 0, zSize = 0;								//the size of the volume in x, y and z direction, is calculated when sinogram is created
		size_t imageWidth = 0, imageHeight = 0;								//stores the height and width of the images in the sinogram
		int imageType;														//assumed type of all the images (taken from the first image)
		double SD = 0;														//the distance of the source to the detector in pixel
		double SO = 0;														//the distance of the source to the object in pixel
		double pixelSize = 0;
		double uOffset = 0, vOffset = 0;									//the offset of the rotation axis in u direction																			//bounds of what will be reconstructed

		//variables that can be set from outside and controls the behaviour of the object
		FilterType filterType = FilterType::RAMLAK;							//holds the frequency filter type that shall be used
		bool emitSignals = true;											//if true the object emits qt signals in certain functions
		size_t crossSectionIndex = 0;										//index for the crossection that is returned in qt signals
		Axis crossSectionAxis = Axis::Z;									//the axis at which the volume is sliced for the cross section
		bool useCuda = true;												//enables or disables the use of cuda
		std::vector<int> activeCudaDevices;									//containing the deviceIds of the gpus that shall be used
		size_t gpuSpareMemory = 200;										//the amount of gpu memory to spare in Mb
		mutable std::atomic<bool> stopActiveProcess{ false };				//is set to true when stop() is called
		double xFrom_float = 0, xTo_float = 1;								//these values control the ROI of the volume that is reconstructed within 0 <= x <= 1
		double yFrom_float = 0, yTo_float = 1;
		double zFrom_float = 0, zTo_float = 1;

		//variables that are only internally used
		Volume<float> volume;												//holds the reconstructed volume
		size_t xFrom = 0, xTo = 0;											//the volume ROI in actual volume coordinates, calculated from the float ROI
		size_t yFrom = 0, yTo = 0;
		size_t zFrom = 0, zTo = 0;
		size_t xMax = 0, yMax = 0, zMax = 0;								//the width, height and depth of the volume ROI that is going to be reconstructed
		mutable std::pair<float, float> minMaxValues;						//stores the brightest and darkes value in all of the sinogram images (for normalisation)
		std::map<int, double> cudaThreadProgress;							//for keeping track of the progress on multithread CUDA execution
		mutable std::mutex exclusiveFunctionsMutex;							//this mutex makes sure certain functions are not executed concurrently
		mutable std::atomic<bool> stopCudaThreads{ false };					//this will be set to true if all cuda threads shall be stopped
		std::string lastErrorMessage;										//if an error in one of the threads occurs, it will be saved here

		//variables for precomputed parts of coordinate transformations
		double worldToVolumeXPrecomputed;
		double worldToVolumeYPrecomputed;
		double worldToVolumeZPrecomputed;
		double volumeToWorldXPrecomputed;
		double volumeToWorldYPrecomputed;
		double volumeToWorldZPrecomputed;
		double imageToMatUPrecomputed;
		double imageToMatVPrecomputed;
		double matToImageUPreprocessed;
		double matToImageVPreprocessed;
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