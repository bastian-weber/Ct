#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void applyHighpassFilter(cv::Mat& img){
	cv::Mat mask = (cv::Mat_<char>(3, 3) <<	 0, 0, 0, 
											-1, 0, 1, 
											 0, 0, 0);
	cv::filter2D(img, img, img.depth(), mask, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
}

void applyRampFilter(cv::Mat& img){
	//CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.channels() == 1);
	
	const double minAttenuation = 1;
	const double maxAttenuation = 0;

	int r = img.rows;
	int c = img.cols;

	double centerColumn = (double)img.cols / 2;
	uchar* ptr;
	for(int i = 0; i < r; ++i){
		ptr = img.ptr<uchar>(i);
		double actualColumn = -centerColumn;
		for(int j = 0; j < c; ++j){
			ptr[j] = uchar((double)ptr[j] * ((abs(actualColumn) / centerColumn) * minAttenuation + ((centerColumn - abs(actualColumn) )/centerColumn) * maxAttenuation));
			++actualColumn;
		}
	}
}

int main(){
	cv::Mat image = cv::imread("../sourcefiles/data/uncompressed.tif", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Image", image);
	applyHighpassFilter(image);
	imshow("Filtered", image);
	applyRampFilter(image);
	imshow("Ramp", image);
	cv::waitKey(0);
}