#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//Applies a filter mask to a 32bit float image
//Mind that the values after applying the filter may be greater than 1 and negative
void applyHighpassFilter(cv::Mat& img){
	CV_Assert(img.depth() == CV_32F);
	cv::Mat mask = (cv::Mat_<char>(3, 3) <<	-1, 0, 1, 
											-2, 0, 2, 
											-1, 0, 1);
	cv::filter2D(img, img, img.depth(), mask, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
}

//Applies a ramp filter to a 32bit float image with 1 channel; weights the center less than the borders (in horizontal direction)
void applyRampFilter(cv::Mat& img){
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() == CV_32F);

	const double minAttenuation = 1;	//the values for the highest and lowest weight
	const double maxAttenuation = 0;

	int r = img.rows;
	int c = img.cols;

	double centerColumn = (double)img.cols / 2;
	double factor;
	double inverseFactor;
	float* ptr;
	for(int i = 0; i < r; ++i){
		ptr = img.ptr<float>(i);
		for(int j = 0; j < c; ++j){
			factor = abs(j - centerColumn) / centerColumn;
			inverseFactor = 1 - factor;
			ptr[j] = ptr[j] * (factor * minAttenuation + inverseFactor * maxAttenuation);
		}
	}
}

//converts an image to 32 bit float
//only unsigned types are allowed as input
void convertTo32bit(cv::Mat& img){
	CV_Assert(img.depth() == CV_8U || img.depth() == CV_16U || img.depth() == CV_32F);
	if (img.depth() == CV_8U){
		img.convertTo(img, CV_32F, 1.0 / (float)pow(2, 8));
	} else if (img.depth() == CV_16U){
		img.convertTo(img, CV_32F, 1.0 / (float)pow(2, 16));
	}
}

int main(){
	cv::Mat image = cv::imread("../sourcefiles/data/uncompressed16.tif", CV_LOAD_IMAGE_ANYDEPTH);
	convertTo32bit(image);
	imshow("Image", image);
	applyHighpassFilter(image);
	imshow("Filtered", image);
	applyRampFilter(image);
	imshow("Ramp", image);
	cv::waitKey(0);
}