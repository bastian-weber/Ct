#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <regex>
#include <fstream>												//used for the output (writing binary file)

#include <opencv2/core/core.hpp>								//core functionality of OpenCV
#include <opencv2/highgui/highgui.hpp>							//GUI functionality of OpenCV (display images etc)
#include <opencv2/imgproc/imgproc.hpp>							//image processing functionality of OpenCV (filter masks etc)
#include "dirent.h"												//library for accessing the filesystem
#include <cmath>												

class CtVolume{
public:
	//functions		
	CtVolume();													//constructor 1
	CtVolume(std::string path);									//constructor 2
	void sinogramFromImages(std::string path);					//creates a sinogram from the images in the specified path
	void displaySinogram() const;								//lets the user scroll through the images in the sinogram	
	void reconstructVolume();									//reconstructs the 3d-volume from the sinogram
	void saveVolumeToBinaryFile(std::string filename) const;	//saves the reconstructed volume to a binary file
private:
	//variables		
	std::vector<cv::Mat> _sinogram;								//here the images are stored
	std::vector<std::vector<std::vector<float>>> _volume;		//holds the reconstructed volume
	mutable int _currentlyDisplayedImage;						//holds the index of the image that is currently being displayed								
	mutable int _xSize;											//the size of the volume in x, y and z direction, is calculated
	mutable int _ySize;											//when sinogram is created
	mutable int _zSize;
	mutable int _imageWidth;									//stores the height and width of the images in the sinogram
	mutable int _imageHeight;									//assigned when sinogram is created
	//functions					
	void handleKeystrokes() const;								//handles the forward and backward arrow keys when sinogram is displayed
	void convertTo32bit(cv::Mat& img) const;					//converts an image to 32bit float
	void applyRampFilter(cv::Mat& img) const;					//applies the ramp filter to an image
	void applyHighpassFilter(cv::Mat& img) const;				//applies the highpass filter to an image

	//used for algorithm 1
	//double A(double u, double s, double D) const;				//helper function for the volume reconstruction
	//double W(double s, double D, double deltaBeta) const;		//helper function for the volume reconstruction
	
	//used for algorithm 2
	double W(double D, double u, double v) const;

	int worldToVolumeX(int xCoord) const;						//coordinate transformations from the coordinates of the vector to
	int worldToVolumeY(int yCoord) const;						//the coordinates of the "world" and the other way around
	int worldToVolumeZ(int zCoord) const;
	int volumeToWorldX(int xCoord) const;
	int volumeToWorldY(int yCoord) const;
	int volumeToWorldZ(int zCoord) const;
	int imageToMatU(int uCoord)const;							//coordinate transformations from the coordinates of the image
	int imageToMatV(int vCoord)const;							//to the coordinates of the saved matrix (always starting at 0)
};