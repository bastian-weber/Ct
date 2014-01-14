#include "CtVolume.h"


//============================================== PUBLIC ==============================================\\

//constructor
CtVolume::CtVolume() :_currentlyDisplayedImage(0){
	//empty
}

CtVolume::CtVolume(std::string path) : _currentlyDisplayedImage(0){
	sinogramFromImages(path);
}

void CtVolume::sinogramFromImages(std::string path){
	//delete the contents of the sinogram
	_sinogram.clear();
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
		_sinogram.resize(count);
		int cnt = 0;
		while ((file = readdir(dir)) != NULL){
			if (std::regex_match(file->d_name, expression)){
				_sinogram[cnt] = cv::imread(path + "/" + file->d_name, CV_LOAD_IMAGE_ANYDEPTH);

				//create some output
				if (!_sinogram[cnt].data){
					std::cout << "Error loading the image " << file->d_name << std::endl;
					return;
				} else{
					std::cout << "Loaded " << file->d_name << std::endl;
				}
				++cnt;
			}
		}
		closedir(dir);
		//now save the resulting size of the volume in member variables (needed for coodinate conversions later)
		if (_sinogram.size() > 0){
			_imageWidth = _sinogram[0].cols;
			_imageHeight = _sinogram[0].rows;
			//Axes: width = x, breadth = y, height = z
			_xSize = _imageWidth;
			_ySize = _imageWidth;
			_zSize = _imageHeight;
			for (std::vector<cv::Mat>::iterator it = _sinogram.begin(); it != _sinogram.end(); ++it){
				convertTo32bit(*it);
				applyHighpassFilter(*it);
				applyRampFilter(*it);
			}	
		}
	} else{
		std::cout << "Could not open the specified directory." << std::endl;
	}
	//now convert them to 32bit float and apply the filters

}

void CtVolume::displaySinogram() const{
	if (_sinogram.size() > 0){
		if (_currentlyDisplayedImage < 0)_currentlyDisplayedImage = _sinogram.size() - 1;
		if (_currentlyDisplayedImage >= _sinogram.size())_currentlyDisplayedImage = 0;
		imshow("Image", _sinogram[_currentlyDisplayedImage]);
		handleKeystrokes();
	} else{
		std::cout << "Could not display sinogram, it is empty." << std::endl;
	}
}

//void CtVolume::reconstructVolume(){
//	if (_sinogram.size() > 0){
//		//resize the volume to the correct size
//		_volume.clear();
//		_volume = std::vector<std::vector<std::vector<float>>>(_xSize, std::vector<std::vector<float>>(_ySize, std::vector<float>(_zSize)));
//		//fill thy volume
//		for (int x = volumeToWorldX(0); x < volumeToWorldX(_xSize); ++x){
//			for (int y = volumeToWorldY(0); y < volumeToWorldY(_ySize); ++y){
//				for (int z = volumeToWorldZ(0); z < volumeToWorldX(_zSize); ++z){
//					if (abs(x) < (_xSize / 2) && abs(y) < (_ySize / 2) && abs(z) < (_zSize / 2)){
//						_volume[worldToVolumeX(x)][worldToVolumeY(y)][worldToVolumeZ(z)] = 1;
//					} else{
//						_volume[worldToVolumeX(x)][worldToVolumeY(y)][worldToVolumeZ(z)] = 0;
//					}
//				}
//			}
//		}
//		std::cout << "Volume successfully reconstructed" << std::endl;
//	} else{
//		std::cout << "Volume was not reconstructed, because the sinogram seems to be empty. Please load some images first." << std::endl;
//	}
//}

//void CtVolume::reconstructVolume(){
//	if (_sinogram.size() > 0){
//		//resize the volume to the correct size
//		_volume.clear();
//		_volume = std::vector<std::vector<std::vector<float>>>(_xSize, std::vector<std::vector<float>>(_ySize, std::vector<float>(_zSize)));
//		//fill the volume
//		double deltaBeta = 360.0 / (double)_sinogram.size();
//		double D = 1;
//		for (int x = volumeToWorldX(0); x < volumeToWorldX(_xSize); ++x){
//			std::cout << x << std::endl;
//			for (int y = volumeToWorldY(0); y < volumeToWorldY(_ySize); ++y){
//				for (int z = volumeToWorldZ(0); z < volumeToWorldX(_zSize); ++z){
//					double sum = 0;
//					for (int projection = 0; projection < _sinogram.size(); ++projection){
//						double beta = deltaBeta*projection;
//						double beta_rad = (beta / 180)*M_PI;
//						double t = x*cos(beta_rad) - y*sin(beta_rad);
//						double s = (-1)*x*sin(beta_rad) - y*cos(beta_rad);
//						double u = A(t, s, D);
//						double v = A(z, s, D);
//						//rounding
//						u = floor(u + 0.5);
//						v = floor(v + 0.5);
//						double weight = W(s, D, deltaBeta);
//						if (imageToMatU(u) < _imageHeight && imageToMatU(u) >= 0 && imageToMatV(v) < _imageWidth && imageToMatV(v) >= 0){
//							sum += weight*_sinogram[projection].at<float>(imageToMatU(u), imageToMatV(v));
//						}
//					}
//					_volume[worldToVolumeX(x)][worldToVolumeY(y)][worldToVolumeZ(z)] = sum;
//				}
//			}
//		}
//		std::cout << "Volume successfully reconstructed" << std::endl;
//	} else{
//		std::cout << "Volume was not reconstructed, because the sinogram seems to be empty. Please load some images first." << std::endl;
//	}
//}

void CtVolume::reconstructVolume(){
	if (_sinogram.size() > 0){
		//resize the volume to the correct size
		_volume.clear();
		_volume = std::vector<std::vector<std::vector<float>>>(_xSize, std::vector<std::vector<float>>(_ySize, std::vector<float>(_zSize)));
		//fill the volume
		double deltaBeta = 360.0 / (double)_sinogram.size();
		double D = 1;
		for (int x = volumeToWorldX(0); x < volumeToWorldX(_xSize); ++x){
			std::cout << x << std::endl;
			for (int y = volumeToWorldY(0); y < volumeToWorldY(_ySize); ++y){
				for (int z = volumeToWorldZ(0); z < volumeToWorldX(_zSize); ++z){
					double sum = 0;
					for (int projection = 0; projection < _sinogram.size(); ++projection){
						double beta = deltaBeta*projection;
						double beta_rad = (beta / 180)*M_PI;
						double t = -(1)*x*sin(beta_rad) + y*cos(beta_rad);
						double s = x*cos(beta_rad) + y*sin(beta_rad);
						double u = (t*D) / (D - s);
						double v = (z*D) / (D - s);
						//rounding
						u = floor(u + 0.5);
						v = floor(v + 0.5);
						double weight = W(D, u, v);
						//std::cout << u << "  " << v << std::endl;
						//system("pause");
						if (imageToMatU(u) < _imageHeight && imageToMatU(u) >= 0 && imageToMatV(v) < _imageWidth && imageToMatV(v) >= 0){
							sum += weight*_sinogram[projection].at<float>(imageToMatU(u), imageToMatV(v));
						}
					}
					_volume[worldToVolumeX(x)][worldToVolumeY(y)][worldToVolumeZ(z)] = sum;
				}
			}
		}
		std::cout << "Volume successfully reconstructed" << std::endl;
	} else{
		std::cout << "Volume was not reconstructed, because the sinogram seems to be empty. Please load some images first." << std::endl;
	}
}

void CtVolume::saveVolumeToBinaryFile(std::string filename) const{
	if (_volume.size() > 0 && _volume[0].size() > 0 && _volume[0][0].size() > 0){
		//open a filestream to the specified filename
		std::ofstream stream(filename, std::ios::out | std::ios::binary);
		if (!stream.is_open()){
			std::cout << "Could not open the filestream. Maybe your path does not exist. No files were written." << std::endl;
		} else{
			//iterate through the volume
			for (int x = 0; x < _xSize; ++x){
				for (int y = 0; y < _ySize; ++y){
					for (int z = 0; z < _zSize; ++z){
						//save one float of data
						stream.write((char*)&_volume[x][y][z], sizeof(float));
					}
				}
			}
			stream.close();
			std::cout << "Volume successfully saved" << std::endl;
		}
	} else{
		std::cout << "Did not save the volume, because it appears to be empty." << std::endl;
	}
}

//============================================== PRIVATE ==============================================\\

void CtVolume::handleKeystrokes() const{
	int key = cv::waitKey(0);				//wait for a keystroke forever
	if (key == 2424832){					//left arrow key
		++_currentlyDisplayedImage;
	} else if (key == 2555904){				//right arrow key
		--_currentlyDisplayedImage;
	} else if (key == 27){					//escape key
		return;
	} else if (key == -1){					//if no key was pressed (meaning the window was closed)
		return;
	}
	if (_currentlyDisplayedImage < 0)_currentlyDisplayedImage = _sinogram.size() - 1;
	if (_currentlyDisplayedImage >= _sinogram.size())_currentlyDisplayedImage = 0;
	imshow("Image", _sinogram[_currentlyDisplayedImage]);
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

//double CtVolume::A(double u, double s, double D) const{
//	return (D*u) / (D - s);
//}
//
//double CtVolume::W(double s, double D, double deltaBeta) const{
//	return (D*D*deltaBeta) / (2 * pow(D - s, 2));
//}

double CtVolume::W(double D, double u, double v) const{
	return D / sqrt(D*D + u*u + v*v);
}



int CtVolume::worldToVolumeX(int xCoord) const{
	return xCoord + (_xSize / 2);
}

int CtVolume::worldToVolumeY(int yCoord) const{
	return yCoord + (_ySize / 2);
}

int CtVolume::worldToVolumeZ(int zCoord) const{
	return zCoord + (_zSize / 2);
}

int CtVolume::volumeToWorldX(int xCoord) const{
	return xCoord - (_xSize / 2);
}

int CtVolume::volumeToWorldY(int yCoord) const{
	return yCoord - (_ySize / 2);
}

int CtVolume::volumeToWorldZ(int zCoord) const{
	return zCoord - (_zSize / 2);
}

int CtVolume::imageToMatU(int uCoord)const{
	return uCoord + (_imageHeight / 2);
}

int CtVolume::imageToMatV(int vCoord)const{
	return vCoord + (_imageWidth / 2);
}
