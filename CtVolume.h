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

//FFTW
#include <fftw3.h>															//FFTW - provides fast fourier transform functionality	

//Qt
#include <QtCore/QtCore>

//for std::numeric_limits<std::streamsize>::max()
#undef max

namespace ct {

	//struct for storing one projection
	struct Projection{
		Projection();
		Projection(cv::Mat image, double angle, double heightOffset);			//Constructor
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
		CtVolume(std::string csvFile,
				 FilterType filterType = FilterType::RAMLAK);
		//getters
		Projection getProjectionAt(size_t index) const;
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
		void sinogramFromImages(std::string csvFile,							//creates a sinogramm out of images specified in csvFile, filterType specifies the prefilter
								FilterType filterType = FilterType::RAMLAK);
		void displaySinogram(bool normalize = false) const;						//lets the user scroll through the images in the sinogram, set normalize for normalizing the gray values	
		void setVolumeBounds(double xFrom, 
							 double xTo, 
							 double yFrom, 
							 double yTo,
							 double zFrom, 
							 double zTo);
		void reconstructVolume();												//reconstructs the 3d-volume from the sinogram
		void saveVolumeToBinaryFile(std::string filename) const;				//saves the reconstructed volume to a binary file
		void stop();															//should stop the operation that's currently running (either preprocessing, reconstruction or saving)
		void setCrossSectionIndex(size_t index);
		void setCrossSectionAxis(Axis axis);
		void setEmitSignals(bool value);
	private:
		//variables		
		bool _emitSignals;														//if true the object emits qt signals in certain functions
		size_t _crossSectionIndex;												//index for the crossection that is returned in qt signals
		Axis _crossSectionAxis;
		mutable std::atomic<bool> _stop;
		std::vector<Projection> _sinogram;										//here the images are stored
		std::vector<std::vector<std::vector<float>>> _volume;					//holds the reconstructed volume
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
		mutable bool _minMaxCaclulated;											//specifies if the min/max values have been calculated for the current sinogram
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
		bool readImages(std::ifstream& csvStream, std::string path);
		void makeHeightOffsetRelative();
		void correctAngleDirection(std::string rotationDirection);
		std::pair<float, float> getSinogramMinMaxIntensity() const;				//returns the highest and lowest density value out of all images in the sinogram
		cv::Mat normalizeImage(cv::Mat const& image,							//returns a new image which is a version of the old image that is normalized by min and max value
							   float minValue,
							   float maxValue) const;
		void handleKeystrokes(bool normalize) const;							//handles the forward and backward arrow keys when sinogram is displayed
		void imagePreprocessing(FilterType filterType);							//applies the necessary filters to the images prior to the reconstruction
		static void convertTo32bit(cv::Mat& img);								//converts an image to 32bit float
		void applyWeightingFilter(cv::Mat& img) const;							//applies the ramp filter to an image
		void applyFeldkampWeight(cv::Mat& image) const;
		static void applyFourierFilter(cv::Mat& image,							//applies a filter in the frequency domain (only in u direction)
									   FilterType type);
		static void applyFourierFilterOpenCV(cv::Mat& image,
											 FilterType type);
		static void applyLogScaling(cv::Mat& image);							//applies a logarithmic scaling to an image
		static double logFunction(double x);									//the actual log function used by applyLogScaling
		static double ramLakWindowFilter(double n, double N);					//Those functions return the scaling coefficients for the
		static double sheppLoganWindowFilter(double n, double N);
		static double hannWindowFilter(double n, double N);						//fourier filters for each n out of N
		static void applyFourierHighpassFilter2D(cv::Mat& image);				//applies a highpass filter in the frequency domain (2D) (not used)
		void reconstructionCore();												//does the actual reconstruction
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