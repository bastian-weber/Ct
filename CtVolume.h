#include <iostream>
#include <vector>
#include <regex>
#include <opencv2/core/core.hpp>					//core functionality of OpenCV
#include <opencv2/highgui/highgui.hpp>				//GUI functionality of OpenCV (display images etc)
#include <opencv2/imgproc/imgproc.hpp>				//image processing functionality of OpenCV (filter masks etc)
#include "dirent.h"									//library for accessing the filesystem

class CtVolume{
public:
	//functions
	CtVolume();										//constructor 1
	CtVolume(std::string path);						//constructor 2
	void sinogramFromImages(std::string path);		//creates a sinogram from the images in the specified path
	void displaySinogram();							//lets the user scroll through the images in the sinogram	
private:
	//variables
	std::vector<cv::Mat> sinogram;					//here the images are stored
	int currentlyDisplayedImage = 0;				//holds the index of the image that is currently being displayed
	//functions			
	void handleKeystrokes();						//handles the forward and backward arrow keys when sinogram is displayed
	void convertTo32bit(cv::Mat& img);				//converts an image to 32bit float
	void applyRampFilter(cv::Mat& img);				//applies the ramp filter to an image
	void applyHighpassFilter(cv::Mat& img);			//applies the highpass filter to an image
};