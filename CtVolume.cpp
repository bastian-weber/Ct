#include "CtVolume.h"

//constructor of data type struct "projection"

Projection::Projection(){}

Projection::Projection(cv::Mat image, double angle) :image(image), angle(angle){
	//empty
}

//============================================== PUBLIC ==============================================\\

//constructor
CtVolume::CtVolume() :_currentlyDisplayedImage(0){
	//empty
}

CtVolume::CtVolume(std::string folderPath, std::string csvPath) : _currentlyDisplayedImage(0){
	sinogramFromImages(folderPath, csvPath);
}

void CtVolume::sinogramFromImages(std::string folderPath, std::string csvPath){
	//delete the contents of the sinogram
	_sinogram.clear();
	//read the angles from the csv file
	std::vector<double> angles;
	if (readCSV(csvPath, angles)){
		//load the files in a sinogram
		DIR* dir;
		struct dirent* file;
		//only load tif files
		std::regex expression("^.*\.tif$");
		if ((dir = opendir(folderPath.c_str())) != NULL){
			//first count the files (so no reallocation of the vector size is necessary)
			int count = 0;
			while ((file = readdir(dir)) != NULL){
				if (std::regex_match(file->d_name, expression)){
					++count;
				}
			}
			rewinddir(dir);

			//check if amount of provided angle values and amount of files match
			if (count == angles.size()){
				//now load the files
				_sinogram.resize(count);
				int cnt = 0;
				while ((file = readdir(dir)) != NULL){
					if (std::regex_match(file->d_name, expression)){
						_sinogram[cnt] = Projection(cv::imread(folderPath + "/" + file->d_name, CV_LOAD_IMAGE_UNCHANGED), angles[cnt]);

						//create some output
						if (!_sinogram[cnt].image.data){
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
					_imageWidth = _sinogram[0].image.cols;
					_imageHeight = _sinogram[0].image.rows;
					//Axes: breadth = x, width = y, height = z
					_xSize = _imageWidth;
					_ySize = _imageWidth;
					_zSize = _imageHeight;
					//now convert them to 32bit float and apply the filters
					for (std::vector<Projection>::iterator it = _sinogram.begin(); it != _sinogram.end(); ++it){
						convertTo32bit(it->image);
						applyHighpassFilter(it->image);
						applyRampFilter(it->image);
					}
				}
			} else{
				std::cout << "The amount of angles provided in the CSV file does not match the amount of images in the folder.";
			}
		} else{
			std::cout << "Could not open the specified directory." << std::endl;
		}

	}
}

void CtVolume::displaySinogram() const{
	if (_sinogram.size() > 0){
		if (_currentlyDisplayedImage < 0)_currentlyDisplayedImage = _sinogram.size() - 1;
		if (_currentlyDisplayedImage >= _sinogram.size())_currentlyDisplayedImage = 0;
		imshow("Projections", _sinogram[_currentlyDisplayedImage].image);
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
//					if (sqrt(pow((double)x, 2) + pow((double)y, 2) + pow((double)z, 2)) <= 50){
//						_volume[worldToVolumeX(x)][worldToVolumeY(y)][worldToVolumeZ(z)] = 10000000;
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

void CtVolume::reconstructVolume(ThreadingType threading){
	if (_sinogram.size() > 0){
		//resize the volume to the correct size
		_volume.clear();
		_volume = std::vector<std::vector<std::vector<float>>>(_xSize, std::vector<std::vector<float>>(_ySize, std::vector<float>(_zSize)));
		//mesure time
		clock_t start = clock();
		//fill the volume
		double deltaBeta = (2*M_PI)/(double)_sinogram.size();
		double D = 999;

		if (threading == MULTITHREADED){
			auto thread1 = std::async(std::launch::async, &CtVolume::reconstructionThread, this, cv::Point3i(0, 0, 0), cv::Point3i(_xSize/2, _ySize/2, _zSize/2), D, true);
			auto thread2 = std::async(std::launch::async, &CtVolume::reconstructionThread, this, cv::Point3i(_xSize / 2, 0, 0), cv::Point3i(_xSize, _ySize / 2, _zSize / 2), D, false);
			auto thread3 = std::async(std::launch::async, &CtVolume::reconstructionThread, this, cv::Point3i(0, _ySize / 2, 0), cv::Point3i(_xSize / 2, _ySize, _zSize / 2), D, false);
			auto thread4 = std::async(std::launch::async, &CtVolume::reconstructionThread, this, cv::Point3i(_xSize / 2, _ySize / 2, 0), cv::Point3i(_xSize, _ySize, _zSize / 2), D, false);
			auto thread5 = std::async(std::launch::async, &CtVolume::reconstructionThread, this, cv::Point3i(0, 0, _zSize / 2), cv::Point3i(_xSize / 2, _ySize / 2, _zSize), D, false);
			auto thread6 = std::async(std::launch::async, &CtVolume::reconstructionThread, this, cv::Point3i(_xSize / 2, 0, _zSize / 2), cv::Point3i(_xSize, _ySize / 2, _zSize), D, false);
			auto thread7 = std::async(std::launch::async, &CtVolume::reconstructionThread, this, cv::Point3i(0, _ySize / 2, _zSize / 2), cv::Point3i(_xSize / 2, _ySize, _zSize), D, false);
			auto thread8 = std::async(std::launch::async, &CtVolume::reconstructionThread, this, cv::Point3i(_xSize / 2, _ySize / 2, _zSize / 2), cv::Point3i(_xSize, _ySize, _zSize), D, false);
			thread1.get();
			thread2.get();
			thread3.get();
			thread4.get();
			thread5.get();
			thread6.get();
			thread7.get();
			thread8.get();
		} else{
			reconstructionThread(cv::Point3i(0, 0, 0), cv::Point3i(_xSize, _ySize, _zSize), D, true);
		}

		//mesure time
		clock_t end = clock();
		std::cout << "Volume successfully reconstructed in " << (double)(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
	} else{
		std::cout << "Volume was not reconstructed, because the sinogram seems to be empty. Please load some images first." << std::endl;
	}
}

void CtVolume::reconstructionThread(cv::Point3i lowerBounds, cv::Point3i upperBounds, double D, bool consoleOutput){
	double imageLowerBoundU = matToImageU(0);
	double imageUpperBoundU = matToImageU(_imageWidth);
	double imageLowerBoundV = matToImageV(0);
	double imageUpperBoundV = matToImageV(_imageHeight);
	for (int x = volumeToWorldX(lowerBounds.x); x < volumeToWorldX(upperBounds.x); ++x){
		if (consoleOutput){
			std::cout << "\r" << "Progress: " << ceil((worldToVolumeX(x) - lowerBounds.x) / (upperBounds.x - lowerBounds.x) * 100 + 0.5) << "%";
		}
		for (int y = volumeToWorldY(lowerBounds.y); y < volumeToWorldY(upperBounds.y); ++y){
			for (int z = volumeToWorldZ(lowerBounds.z); z < volumeToWorldZ(upperBounds.z); ++z){

				////testing
				//x = 50;
				//y = 40;
				//z = 10;
				////testing

				double sum = 0;
				for (int projection = 0; projection < _sinogram.size(); ++projection){
					double beta_rad = (_sinogram[projection].angle/180.0) * M_PI;
					double t = (-1)*(double)x*sin(beta_rad) + (double)y*cos(beta_rad);
					double s = (double)x*cos(beta_rad) + (double)y*sin(beta_rad);
					double u = (t*D) / (D - s);
					double v = ((double)z*D) / (D - s);


					if (u > imageLowerBoundU && u < imageUpperBoundU && v > imageLowerBoundV && v < imageUpperBoundV){

						u = imageToMatU(u);
						v = imageToMatV(v);

						////testing
						//_sinogram[projection].image.at<float>(v, u) = 1;
						//imshow("Projections", _sinogram[projection].image);
						//cvWaitKey(0);
						////testing

						double u0 = floor(u);
						double u1 = ceil(u);
						double v0 = floor(v);
						double v1 = ceil(v);
						if (u0 < _imageWidth && u0 >= 0 && u1 < _imageWidth && u1 >= 0 && v0 < _imageHeight && v0 >= 0 && v1 < _imageHeight && v1 >= 0){
							float u0v0 = _sinogram[projection].image.at<float>(v0, u0);
							float u1v0 = _sinogram[projection].image.at<float>(v0, u1);
							float u0v1 = _sinogram[projection].image.at<float>(v1, u0);
							float u1v1 = _sinogram[projection].image.at<float>(v1, u1);
							sum += bilinearInterpolation(u - u0, v - v0, u0v0, u1v0, u0v1, u1v1);
						}
					}
				}
				_volumeMutex.lock();
				_volume[worldToVolumeX(x)][worldToVolumeY(y)][worldToVolumeZ(z)] = sum;
				_volumeMutex.unlock();
			}
		}
	}
	std::cout << "Thread finished" << std::endl;
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

bool CtVolume::readCSV(std::string filename, std::vector<double>& result) const{
	result.clear();
	std::ifstream stream(filename.c_str(), std::ios::in);
	if (!stream.good()){
		std::cerr << "Could not open CSV file - terminating" << std::endl;
		return false;
	}
	//count the lines in the file
	int itemCount = std::count(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>(), '\n') + 1;
	if (itemCount != 0){
		result.resize(itemCount);
		//go back to the beginning
		stream.seekg(std::ios::beg);
		//now read file contents
		std::stringstream strstr;
		std::string line;
		int cnt = 0;
		while (!stream.eof()){
			std::getline(stream, line);
			strstr = std::stringstream(line);
			strstr >> result[cnt];
			++cnt;
		}
		return true;
	} else{
		std::cout << "CSV file seems to be empty - terminating" << std::endl;
		return false;
	}
}

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
	imshow("Image", _sinogram[_currentlyDisplayedImage].image);
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

float CtVolume::bilinearInterpolation(double u, double v, float u0v0, float u1v0, float u0v1, float u1v1) const{
	//the two interpolations on the u axis
	double v0 = (1 - u)*u0v0 + u*u1v0;
	double v1 = (1 - u)*u0v1 + u*u1v1;
	//interpolation on the v axis between the two u-interpolated values
	return (1 - v)*v0 + v*v1;
}

double CtVolume::W(double D, double u, double v) const{
	return D / sqrt(D*D + u*u + v*v);
}



double CtVolume::worldToVolumeX(double xCoord) const{
	return xCoord + ((double)_xSize / 2.0);
}

double CtVolume::worldToVolumeY(double yCoord) const{
	return yCoord + ((double)_ySize / 2.0);
}

double CtVolume::worldToVolumeZ(double zCoord) const{
	return zCoord + ((double)_zSize / 2.0);
}

double CtVolume::volumeToWorldX(double xCoord) const{
	return xCoord - ((double)_xSize / 2.0);
}

double CtVolume::volumeToWorldY(double yCoord) const{
	return yCoord - ((double)_ySize / 2.0);
}

double CtVolume::volumeToWorldZ(double zCoord) const{
	return zCoord - ((double)_zSize / 2.0);
}

double CtVolume::imageToMatU(double uCoord)const{
	return uCoord + ((double)_imageWidth / 2.0);
}

double CtVolume::imageToMatV(double vCoord)const{
	return vCoord + ((double)_imageHeight / 2.0);
}

double CtVolume::matToImageU(double uCoord)const{
	return uCoord - ((double)_imageWidth / 2.0);
}

double CtVolume::matToImageV(double vCoord)const{
	return vCoord - ((double)_imageHeight / 2.0);
}
