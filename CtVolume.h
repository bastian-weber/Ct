#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <regex>
#include <fstream>												//used for the output (writing binary file)
#include <cmath>	
#include <ctime>
#include <future>
#include <mutex>

#include <opencv2/core/core.hpp>								//core functionality of OpenCV
#include <opencv2/highgui/highgui.hpp>							//GUI functionality of OpenCV (display images etc)
#include <opencv2/imgproc/imgproc.hpp>							//image processing functionality of OpenCV (filter masks etc)
#include <fftw3.h>												//FFTW - provides fast fourier transform functionality
#include "dirent.h"												//library for accessing the filesystem						

struct Projection{
	Projection();
	Projection(cv::Mat image, double angle);					//Constructor
	cv::Mat image;												
	double angle;
};

class CtVolume{
public:
	//functions		
	CtVolume();													//constructor 1
	CtVolume(std::string folderPath, std::string csvPath);		//constructor 2
	enum ThreadingType{SINGLETHREADED, MULTITHREADED};
	void sinogramFromImages(std::string folderPath,				//creates a sinogram from the images in the specified path
							std::string csvPath);					
	void displaySinogram() const;								//lets the user scroll through the images in the sinogram	
	void reconstructVolume(ThreadingType threading);				//reconstructs the 3d-volume from the sinogram
	void saveVolumeToBinaryFile(std::string filename) const;	//saves the reconstructed volume to a binary file
private:
	//variables		
	std::vector<Projection> _sinogram;							//here the images are stored
	std::vector<std::vector<std::vector<float>>> _volume;		//holds the reconstructed volume
	mutable int _currentlyDisplayedImage;						//holds the index of the image that is currently being displayed								
	mutable int _xSize;											//the size of the volume in x, y and z direction, is calculated
	mutable int _ySize;											//when sinogram is created
	mutable int _zSize;
	mutable int _imageWidth;									//stores the height and width of the images in the sinogram
	mutable int _imageHeight;									//assigned when sinogram is created
	mutable std::mutex _volumeMutex;							//prevents that two threads access the volume simultaneously
	//functions			
	bool readCSV(std::string filename,							//reads the additional information from the csv file
				 std::vector<double>& result) const;				
	void handleKeystrokes() const;								//handles the forward and backward arrow keys when sinogram is displayed
	void convertTo32bit(cv::Mat& img) const;					//converts an image to 32bit float
	void applyRampFilter(cv::Mat& img) const;					//applies the ramp filter to an image
	void applyHighpassFilter(cv::Mat& img) const;				//applies the highpass filter to an image
	void applyFourierHighpassFilter1D(cv::Mat& image) const;		//applies a highpass filter in the frequency domain (only in u direction)
	void applyFourierHighpassFilter2D(cv::Mat& image) const;		//applies a highpass filter in the frequency domain (2D)
	void reconstructionThread(cv::Point3i lowerBounds, 
							  cv::Point3i upperBounds, 
							  double D,
							  bool consoleOutput);
	float bilinearInterpolation(double u,						//interpolates bilinear between those four intensities
								double v,				
								float u0v0, 
								float u1v0, 
								float u0v1, 
								float u1v1) const;
	double W(double D, double u, double v) const;				//weight function for the reconstruction of the volume
	double worldToVolumeX(double xCoord) const;					//coordinate transformations from the coordinates of the vector to
	double worldToVolumeY(double yCoord) const;					//the coordinates of the "world" and the other way around
	double worldToVolumeZ(double zCoord) const;
	double volumeToWorldX(double xCoord) const;
	double volumeToWorldY(double yCoord) const;
	double volumeToWorldZ(double zCoord) const;
	double imageToMatU(double uCoord)const;						//coordinate transformations from the coordinates of the image
	double imageToMatV(double vCoord)const;						//to the coordinates of the saved matrix (always starting at 0)
	double matToImageU(double uCoord)const;						
	double matToImageV(double vCoord)const;	
	int fftCoordToIndex(int coord, int size) const;				//coordinate transformation for the FFT lowpass filtering
};