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
#include <opencv2/imgproc/imgproc.hpp>										//image processing functionality of OpenCV (filter masks etc)

//Qt
#include <QtCore/QtCore>

//for std::numeric_limits<std::streamsize>::max()
#undef max

namespace ct {

	//struct for returning projections
	struct Projection{
		Projection();
		Projection(cv::Mat image, double angle, double heightOffset);
		cv::Mat image;
		double angle;
		double heightOffset;													//for random trajectory
	};

	enum class FilterType{ 
		RAMLAK, 
		SHEPP_LOGAN, 
		HANN
	};

	enum class Axis{
		X,
		Y,
		Z
	};

	//The actual reconstruction class
	class CtVolume : public QObject{
		Q_OBJECT
	public:
		struct CompletionStatus{
			CompletionStatus() : successful(true), userInterrupted(false) { };
			CompletionStatus(bool successful, bool userInterrupted, QString const& errorMessage = QString()) : successful(successful), userInterrupted(userInterrupted), errorMessage(errorMessage) { }
			static CompletionStatus success() { return CompletionStatus(true, false); }
			static CompletionStatus interrupted() { return CompletionStatus(false, true); }
			static CompletionStatus error(QString const& errorMessage) { return CompletionStatus(false, false, errorMessage); }
			bool successful;
			bool userInterrupted;
			QString errorMessage;
		};

		//functions		
		//constructor
		CtVolume();																
		CtVolume(std::string csvFile);
		//getters
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
		cv::Mat getVolumeCrossSection(size_t index) const;
		size_t getCrossSectionIndex() const;
		size_t getCrossSectionSize() const;
		Axis getCrossSectionAxis() const;
		//control functions
		void sinogramFromImages(std::string csvFile);							//creates a sinogramm out of images specified in csvFile								
		void setVolumeBounds(double xFrom, 
							 double xTo, 
							 double yFrom, 
							 double yTo,
							 double zFrom, 
							 double zTo);
		void reconstructVolume(FilterType filterType = FilterType::RAMLAK);		//reconstructs the 3d-volume from the sinogram, filterType specifies the used prefilter
		void saveVolumeToBinaryFile(std::string filename) const;				//saves the reconstructed volume to a binary file
		void stop();															//should stop the operation that's currently running (either preprocessing, reconstruction or saving)
		void setCrossSectionIndex(size_t index);
		void setCrossSectionAxis(Axis axis);
		void setEmitSignals(bool value);
	private:

		//struct for storing one projection internally
		struct Projection{
			Projection();
			Projection(std::string imagePath, double angle, double heightOffset);	//Constructor
			cv::Mat getImage() const;
			ct::Projection getPublicProjection() const;
			std::string imagePath;
			double angle;
			double heightOffset;													//for random trajectory
		};

		//variables		
		std::vector<Projection> _sinogram;										//here the images are stored
		std::vector<std::vector<std::vector<float>>> _volume;					//holds the reconstructed volume
		std::mutex _exclusiveFunctionsMutex;
		bool _emitSignals;														//if true the object emits qt signals in certain functions
		size_t _crossSectionIndex;												//index for the crossection that is returned in qt signals
		Axis _crossSectionAxis;
		mutable std::atomic<bool> _stop;
		mutable int _currentlyDisplayedImage;									//holds the index of the image that is currently being displayed								
		size_t _xSize, _ySize, _zSize;											//the size of the volume in x, y and z direction, is calculated when sinogram is created
		size_t _imageWidth, _imageHeight;;										//stores the height and width of the images in the sinogram
		//bounds of what will be reconstructed
		double _xFrom_float, _xTo_float;
		double _yFrom_float, _yTo_float;
		double _zFrom_float, _zTo_float;
		size_t _xFrom, _xTo;
		size_t _yFrom, _yTo;
		size_t _zFrom, _zTo;
		size_t _xMax, _yMax, _zMax;
		double _SD;																//the distance of the source to the detector in pixel
		double _SO;																//the distance of the source to the object in pixel
		double _pixelSize;
		double _uOffset, _vOffset;												//the offset of the rotation axis in u direction
		mutable std::pair<float, float> _minMaxValues;
		//some precomputed values for the coordinate conversion functions for faster execution
		double _worldToVolumeXPrecomputed;
		double _worldToVolumeYPrecomputed;
		double _worldToVolumeZPrecomputed;
		double _volumeToWorldXPrecomputed;
		double _imageToMatUPrecomputed;
		double _imageToMatVPrecomputed;
		//functions			
		void readParameters(std::ifstream& stream, 
								   std::string& path, 
								   std::string& rotationDirection);
		std::string glueRelativePath(std::string const& basePath, 
									 std::string const& potentialRelativePath);
		bool readImages(std::ifstream& csvStream, std::string path, int imgCnt);
		void makeHeightOffsetRelative();
		void correctAngleDirection(std::string rotationDirection);
		cv::Mat normalizeImage(cv::Mat const& image,							//returns a new image which is a version of the old image that is normalized by min and max value
							   float minValue,
							   float maxValue) const;
		cv::Mat prepareProjection(size_t index, FilterType filterType) const;							//returns the image of the projection at position index preprocessed and converted
		void preprocessImage(cv::Mat& image, FilterType filterType) const;
		static void convertTo32bit(cv::Mat& img);								//converts an image to 32bit float
		void applyFeldkampWeight(cv::Mat& image) const;
		static void applyFourierFilter(cv::Mat& image,
											 FilterType type);
		static void applyLogScaling(cv::Mat& image);							//applies a logarithmic scaling to an image
		static double logFunction(double x);									//the actual log function used by applyLogScaling
		static double ramLakWindowFilter(double n, double N);					//Those functions return the scaling coefficients for the
		static double sheppLoganWindowFilter(double n, double N);
		static double hannWindowFilter(double n, double N);						//fourier filters for each n out of N
		bool reconstructionCore(FilterType filterType = FilterType::RAMLAK);	//does the actual reconstruction, filterType specifies the type of the highpass filter
		static float bilinearInterpolation(double u,							//interpolates bilinear between those four intensities
									double v,
									float u0v0,
									float u1v0,
									float u0v1,
									float u1v1);
		static double W(double D, double u, double v);							//weight function for the reconstruction of the volume
		//coordinate transformation functions
		void updateBoundaries();
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
		static int fftCoordToIndex(int coord, int size);							//coordinate transformation for the FFT lowpass filtering, only used for the 2D highpass filtering, which is currently not used
	signals:	
		void loadingProgress(double percentage) const;
		void loadingFinished(CtVolume::CompletionStatus status = CompletionStatus::success()) const;
		void reconstructionProgress(double percentage, cv::Mat crossSection) const;
		void reconstructionFinished(cv::Mat crossSection, CtVolume::CompletionStatus status = CompletionStatus::success()) const;
		void savingProgress(double percentage) const;
		void savingFinished(CtVolume::CompletionStatus status = CompletionStatus::success()) const;
	};

}

#endif