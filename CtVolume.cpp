#include "CtVolume.h"

//============================================== PUBLIC ==============================================\\

//constructor
CtVolume::CtVolume():currentlyDisplayedImage(0){
	//empty
}

CtVolume::CtVolume(std::string path):currentlyDisplayedImage(0){
	sinogramFromImages(path);
}

void CtVolume::sinogramFromImages(std::string path){
	//delete the contents of the sinogram
	sinogram.clear();
	//load the files in a sinogram
	DIR* dir;
	struct dirent* file;

	//only load tif files
	std::regex expression("^.*\.tif$");
	if ((dir = opendir(path.c_str())) != NULL){
		//first count the files (so no reallocation of the vector size is necessary)
		int count = 0;
		while ((file = readdir(dir)) != NULL){
			if (std::regex_match(file->d_name, expression)){
				++count;
			}
		}
		rewinddir(dir);

		//now load the files
		sinogram.resize(count);
		int cnt = 0;
		while ((file = readdir(dir)) != NULL){
			if (std::regex_match(file->d_name, expression)){
				sinogram[cnt] = cv::imread(path + "/" + file->d_name, CV_LOAD_IMAGE_ANYDEPTH);

				//create some output
				if (!sinogram[cnt].data){
					std::cout << "Error loading the image " << file->d_name << std::endl;
					return;
				} else{
					std::cout << "Loaded " << file->d_name << std::endl;
				}
				++cnt;
			}
		}
		closedir(dir);
	} else{
		std::cout << "Could not open the specified directory." << std::endl;
	}
	//now convert them to 32bit float and apply the filters
	for (std::vector<cv::Mat>::iterator it = sinogram.begin(); it != sinogram.end(); ++it){
		convertTo32bit(*it);
		//applyHighpassFilter(*it);
		//applyRampFilter(*it);
	}
}

void CtVolume::displaySinogram() const{
	if (sinogram.size() > 0){
		if (currentlyDisplayedImage < 0)currentlyDisplayedImage = sinogram.size() - 1;
		if (currentlyDisplayedImage >= sinogram.size())currentlyDisplayedImage = 0;
		imshow("Image", sinogram[currentlyDisplayedImage]);
		handleKeystrokes();
	} else{
		std::cout << "Could not display sinogram, it is empty." << std::endl;
	}
}

//============================================== PRIVATE ==============================================\\

void CtVolume::handleKeystrokes() const{
	int key = cv::waitKey(0);				//wait for a keystroke forever
	if (key == 2424832){					//left arrow key
		++currentlyDisplayedImage;	
	} else if (key == 2555904){				//right arrow key
		--currentlyDisplayedImage;	
	} else if (key == 27){					//escape key
		return;	
	} else if (key == -1){					//if no key was pressed (meaning the window was closed)
		return;
	}
	if (currentlyDisplayedImage < 0)currentlyDisplayedImage = sinogram.size() - 1;
	if (currentlyDisplayedImage >= sinogram.size())currentlyDisplayedImage = 0;
	imshow("Image", sinogram[currentlyDisplayedImage]);
	handleKeystrokes();
}

//converts an image to 32 bit float
//only unsigned types are allowed as input
void CtVolume::convertTo32bit(cv::Mat& img) const{
	CV_Assert(img.depth() == CV_8U || img.depth() == CV_16U || img.depth() == CV_32F);
	if (img.depth() == CV_8U){
		img.convertTo(img, CV_32F, 1.0 / (float)pow(2, 8));
	} else if (img.depth() == CV_16U){
		img.convertTo(img, CV_32F, 1.0 / (float)pow(2, 16));
	}
}

//Applies a ramp filter to a 32bit float image with 1 channel; weights the center less than the borders (in horizontal direction)
void CtVolume::applyRampFilter(cv::Mat& img) const{
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
	for (int i = 0; i < r; ++i){
		ptr = img.ptr<float>(i);
		for (int j = 0; j < c; ++j){
			factor = abs(j - centerColumn) / centerColumn;
			inverseFactor = 1 - factor;
			ptr[j] = ptr[j] * (factor * minAttenuation + inverseFactor * maxAttenuation);
		}
	}
}

//Applies a filter mask to a 32bit float image
//Mind that the values after applying the filter may be greater than 1 and negative
void CtVolume::applyHighpassFilter(cv::Mat& img) const{
	CV_Assert(img.depth() == CV_32F);
	cv::Mat mask = (cv::Mat_<char>(3, 3) << -1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);
	cv::filter2D(img, img, img.depth(), mask, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
}