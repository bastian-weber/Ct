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

CtVolume::CtVolume(std::string folderPath, std::string csvPath, CtVolume::FileType fileType, CtVolume::FilterType filterType) : _currentlyDisplayedImage(0){
	sinogramFromImages(folderPath, csvPath, fileType, filterType);
}

void CtVolume::sinogramFromImages(std::string folderPath, std::string csvPath, CtVolume::FileType fileType, CtVolume::FilterType filterType){
	//delete the contents of the sinogram
	_sinogram.clear();
	//read the angles from the csv file
	std::vector<double> angles;

	if (readCSV(csvPath, angles)){

		//load the files in a sinogram
		DIR* dir;
		struct dirent* file;
		//only load tif files
		std::regex expression;
		switch (fileType){
		case BMP:
			expression = std::regex("^.*\.bmp$", std::regex::icase);
			break;
		case JPG:
			expression = std::regex("^.*\.(jpg|jpeg|jpe)$", std::regex::icase);
			break;
		case JPEG2000:
			expression = std::regex("^.*\.jp2$", std::regex::icase);
			break;
		case PNG:
			expression = std::regex("^.*\.png$", std::regex::icase);
			break;
		case TIF:
			expression = std::regex("^.*\.(tif|tiff)$", std::regex::icase);
			break;
		}
		if ((dir = opendir(folderPath.c_str())) != NULL){
			//first count the files (so no reallocation of the vector size is necessary)
			int count = 0;
			while ((file = readdir(dir)) != NULL){
				if (std::regex_match(file->d_name, expression)){
					++count;
				}
			}
			rewinddir(dir);

			if (count != 0){
				std::cout << count << " files discovered." << std::endl;
			} else{
				std::cout << "No files of the specified image type could be found." << std::endl;
			}

			//check if amount of provided angle values and amount of files match
			if (count == angles.size()){

				//now load the files
				_sinogram.resize(count);
				int cnt = 0;
				int rows;
				int cols;
				while ((file = readdir(dir)) != NULL){
					if (std::regex_match(file->d_name, expression)){

						_sinogram[cnt] = Projection(cv::imread(folderPath + "/" + file->d_name, CV_LOAD_IMAGE_UNCHANGED), angles[cnt]);

						//create some output
						if (!_sinogram[cnt].image.data){
							_sinogram.clear();
							std::cout << "Error loading the image " << file->d_name << std::endl;
							return;
						} else if (_sinogram[cnt].image.channels() != 1){				//make sure it's a one-channel image
							_sinogram.clear();
							std::cout << "Error loading the image " << file->d_name << ", it has not exactly 1 channel." << std::endl;
							return;
						}else{
							//make sure that all images have the same size
							if (cnt == 0){
								rows = _sinogram[cnt].image.rows;
								cols = _sinogram[cnt].image.cols;
							} else{
								if (_sinogram[cnt].image.rows != rows || _sinogram[cnt].image.cols != cols){
									_sinogram.clear();
									std::cout << "Error loading the image " << file->d_name << ", its dimensions differ from the images before." << std::endl;
									return;
								}
							}
							std::cout << "Loaded " << file->d_name << std::endl;
						}

						//convert the image to 32 bit float
						convertTo32bit(_sinogram[cnt].image);
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
					//now apply the filters
					imagePreprocessing(filterType);
				}

			} else{
				std::cout << "The amount of angles provided in the CSV file does not match the amount of images in the folder." << std::endl;
			}
		} else{
			std::cout << "Could not open the specified directory." << std::endl;
		}

	}
}

void CtVolume::displaySinogram(bool normalize) const{
	if (_sinogram.size() > 0){
		if (_currentlyDisplayedImage < 0)_currentlyDisplayedImage = _sinogram.size() - 1;
		if (_currentlyDisplayedImage >= _sinogram.size())_currentlyDisplayedImage = 0;
		if (normalize){
			_minMaxValues = getSinogramMinMaxIntensity();
			cv::Mat normalizedImage = normalizeImage(_sinogram[_currentlyDisplayedImage].image, _minMaxValues.first, _minMaxValues.second);
			imshow("Projections", normalizedImage);
		} else{
			imshow("Projections", _sinogram[_currentlyDisplayedImage].image);
		}
		handleKeystrokes(normalize);
	} else{
		std::cout << "Could not display sinogram, it is empty." << std::endl;
	}
}

void CtVolume::reconstructVolume(ThreadingType threading){
	if (_sinogram.size() > 0){
		//resize the volume to the correct size
		_volume.clear();
		_volume = std::vector<std::vector<std::vector<float>>>(_xSize, std::vector<std::vector<float>>(_ySize, std::vector<float>(_zSize)));
		//mesure time
		clock_t start = clock();
		//fill the volume
		double deltaBeta = (2*M_PI)/(double)_sinogram.size();
		double D = 1000;

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

		//now fill the corners around the cylinder with the lowest density value
		double smallestValue;
		for (int x = 0; x < _xSize; ++x){
			for (int y = 0; y < _ySize; ++y){
				for (int z = 0; z < _zSize; ++z){
					if (x == 0 && y == 0 && z == 0){
						smallestValue = _volume[x][y][z];
					} else if (_volume[x][y][z] < smallestValue){
						smallestValue = _volume[x][y][z];
					}
				}
			}
		}

		for (int x = 0; x < _xSize; ++x){
			for (int y = 0; y < _ySize; ++y){
				if (sqrt(volumeToWorldX(x)*volumeToWorldX(x) + volumeToWorldY(y)*volumeToWorldY(y)) >= ((double)_xSize / 2) - 3){
					for (int z = 0; z < _zSize; ++z){
						_volume[x][y][z] = smallestValue;
					}
				}
			}
		}

		//mesure time
		clock_t end = clock();
		std::cout << "Volume successfully reconstructed (" << (double)(end - start) / CLOCKS_PER_SEC << "s)" << std::endl;
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
		//output percentage
		if (consoleOutput){
			std::cout << "\r" << "Backprojecting: " << floor((worldToVolumeX(x) - lowerBounds.x) / (upperBounds.x - lowerBounds.x) * 100 + 0.5) << "%";
		}
		for (int y = volumeToWorldY(lowerBounds.y); y < volumeToWorldY(upperBounds.y); ++y){
			//if the voxel is inside the reconstructable cylinder
			if (sqrt((double)x*(double)x + (double)y*(double)y) < ((double)_xSize/2) - 3){
				for (int z = volumeToWorldZ(lowerBounds.z); z < volumeToWorldZ(upperBounds.z); ++z){

					//accumulate the densities from all projections
					double sum = 0;
					for (int projection = 0; projection < _sinogram.size(); ++projection){
						double beta_rad = (_sinogram[projection].angle / 180.0) * M_PI;
						double t = (-1)*(double)x*sin(beta_rad) + (double)y*cos(beta_rad);
						double s = (double)x*cos(beta_rad) + (double)y*sin(beta_rad);
						double u = (t*D) / (D - s);
						double v = ((double)z*D) / (D - s);

						//check if it's inside the image (before the coordinate transformation)
						if (u > imageLowerBoundU && u < imageUpperBoundU && v > imageLowerBoundV && v < imageUpperBoundV){

							double weight = W(D, u, v);

								u = imageToMatU(u);
								v = imageToMatV(v);

							//get the 4 surrounding pixels for the bilinear interpolation
							double u0 = floor(u);
							double u1 = ceil(u);
							double v0 = floor(v);
							double v1 = ceil(v);
							//check if all the pixels are inside the image (after the coordinate transformation)
							if (u0 < _imageWidth && u0 >= 0 && u1 < _imageWidth && u1 >= 0 && v0 < _imageHeight && v0 >= 0 && v1 < _imageHeight && v1 >= 0){
								float u0v0 = _sinogram[projection].image.at<float>(v0, u0);
								float u1v0 = _sinogram[projection].image.at<float>(v0, u1);
								float u0v1 = _sinogram[projection].image.at<float>(v1, u0);
								float u1v1 = _sinogram[projection].image.at<float>(v1, u1);
								sum += weight *  bilinearInterpolation(u - u0, v - v0, u0v0, u1v0, u0v1, u1v1);
							}
						}
					}
					_volumeMutex.lock();
					_volume[worldToVolumeX(x)][worldToVolumeY(y)][worldToVolumeZ(z)] = sum;
					_volumeMutex.unlock();
				}
			}
		}
	}
	if (consoleOutput){
		std::cout << std::endl << "Waiting for all threads to finish" << std::endl;
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

std::pair<float, float> CtVolume::getSinogramMinMaxIntensity() const{
	float min = *std::min_element(_sinogram[0].image.begin<float>(), _sinogram[0].image.end<float>());
	float max = *std::max_element(_sinogram[0].image.begin<float>(), _sinogram[0].image.end<float>());
	float tmp;

	for (int i = 1; i < _sinogram.size(); ++i){
		tmp = *std::min_element(_sinogram[i].image.begin<float>(), _sinogram[i].image.end<float>());
		if (tmp < min)min = tmp;
		tmp = *std::max_element(_sinogram[i].image.begin<float>(), _sinogram[i].image.end<float>());
		if (tmp > max)max = tmp;
	}
	return std::make_pair(min, max);
}

cv::Mat CtVolume::normalizeImage(cv::Mat const& image, float minValue, float maxValue) const{
	int R = image.rows;
	int C = image.cols;
	cv::Mat normalizedImage(image.rows, image.cols, CV_32F);

	const float* ptr;
	float* targPtr;
	for (int i = 0; i < R; ++i){
		ptr = image.ptr<float>(i);
		targPtr = normalizedImage.ptr<float>(i);
		for (int j = 0; j < C; ++j){
			targPtr[j] = (ptr[j] - minValue) / (maxValue - minValue);
		}
	}

	return normalizedImage;
}

void CtVolume::handleKeystrokes(bool normalize) const{
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
	if (normalize){
		cv::Mat normalizedImage = normalizeImage(_sinogram[_currentlyDisplayedImage].image, _minMaxValues.first, _minMaxValues.second);
		imshow("Projections", normalizedImage);
	} else{
		imshow("Projections", _sinogram[_currentlyDisplayedImage].image);
	}
	handleKeystrokes(normalize);
}

void CtVolume::imagePreprocessing(CtVolume::FilterType filterType){
	clock_t start = clock();
	for (int i = 0; i < _sinogram.size(); ++i){
		if (i % 20 == 0)std::cout << "\r" << "Preprocessing: " << floor((double)i / (double)_sinogram.size() * 100 + 0.5) << "%";
		applyFourierFilter(_sinogram[i].image, filterType);
		//applyWeightingFilter(_sinogram[i].image);
	}
	std::cout << std::endl;
	clock_t end = clock();
	std::cout << "Preprocessing sucessfully finished (" << (double)(end - start) / CLOCKS_PER_SEC << "s)" << std::endl;
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
void CtVolume::applyWeightingFilter(cv::Mat& img) const{
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
	/*cv::Mat mask = (cv::Mat_<char>(3, 3) << -1, 0, 1,
											-2, 0, 2,
											-1, 0, 1);*/
	cv::Mat mask = (cv::Mat_<char>(3, 3) <<  1,  1, 1,
											 1, -8, 1,
											 1,  1, 1);
	cv::filter2D(img, img, img.depth(), mask, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
}

void CtVolume::applyFourierFilter(cv::Mat& image, CtVolume::FilterType type) const{
	CV_Assert(image.depth() == CV_32F);

	//FFT
	const int R = image.rows;
	const int C = image.cols;

	const int nyquist = (C / 2) + 1;

	float* ptr;
	fftwf_complex *out;
	out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)* nyquist);
	fftwf_plan p;

	for (int row = 0; row < R; ++row){
		ptr = image.ptr<float>(row);
		p = fftwf_plan_dft_r2c_1d(C, ptr, out, FFTW_ESTIMATE);
		fftwf_execute(p);
		fftwf_destroy_plan(p);

		for (int i = 0; i < nyquist; ++i){
			out[i][0] = out[i][0] / C;
			out[i][1] = out[i][1] / C;
		}

		//removing the low frequencies
		double factor;
		for (int column = 0; column < nyquist; ++column){
			switch (type){
			case RAMLAK:
				factor = ramLakWindowFilter(column, nyquist);
				break;
			case HANN:
				factor = hannWindowFilter(column, nyquist);
				break;
			case RECTANGLE:
				factor = rectangleWindowFilter(column, nyquist);
				break;
			}
			out[column][0] *= factor;
			out[column][1] *= factor;
		}

		//inverse
		p = fftwf_plan_dft_c2r_1d(C, out, ptr, FFTW_ESTIMATE);
		fftwf_execute(p);
		fftwf_destroy_plan(p);
	}

	fftwf_free(out);
}

double CtVolume::ramLakWindowFilter(double n, double N) const{
	return (double)n / (double)N;
}

double CtVolume::hannWindowFilter(double n, double N) const{
	return ramLakWindowFilter(n, N) * 0.5*(1 + cos((2 * M_PI * (double)n) / ((double)N * 2)));
}

double CtVolume::rectangleWindowFilter(double n, double N) const{
	double cutoffRatio = 0.09;		//defining the width of the filter rectangle, should be in interval [0,1]
	if (n <= cutoffRatio*N){
		return 0;
	}
	return 1;
}

void CtVolume::applyFourierHighpassFilter2D(cv::Mat& image) const{
	CV_Assert(image.depth() == CV_32F);

	//FFT
	const int R = image.rows;
	const int C = image.cols;

	std::vector<std::vector<std::complex<double>>> result(R, std::vector<std::complex<double>>(C));
	float *in;
	in = (float*)fftwf_malloc(sizeof(float)* R * C);
	int k = 0;
	float* ptr;
	for (int row = 0; row < R; ++row){
		ptr = image.ptr<float>(row);
		for (int column = 0; column < C; ++column){
			in[k] = ptr[column];
			++k;
		}
	}
	fftwf_complex *out;
	int nyquist = (C / 2) + 1;
	out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)* R * C);
	fftwf_plan p;
	p = fftwf_plan_dft_r2c_2d(R, C, in, out, FFTW_ESTIMATE);
	fftwf_execute(p);
	fftwf_destroy_plan(p);

	k = 0;
	for (int row = 0; row < R; ++row){
		for (int column = 0; column < nyquist; ++column){
			out[k][0] = out[k][0] / (R*C);
			out[k][1] = out[k][1] / (R*C);
			++k;
		}
	}

	//removing the low frequencies

	double cutoffRatio = 0.1;
	int uCutoff = (cutoffRatio / 2.0)*C;
	int vCutoff = (cutoffRatio / 2.0)*R;
	for (int row = -vCutoff; row <= vCutoff; ++row){
		for (int column = 0; column <= uCutoff; ++column){
			out[fftCoordToIndex(row, R)*nyquist + column][0] = 0;
			out[fftCoordToIndex(row, R)*nyquist + column][1] = 0;
		}
	}

	//reverse FFT

	p = fftwf_plan_dft_c2r_2d(R, C, out, in, FFTW_ESTIMATE);
	fftwf_execute(p);
	fftwf_destroy_plan(p);
	k = 0;
	for (int row = 0; row < R; ++row){
		ptr = image.ptr<float>(row);
		for (int column = 0; column < C; ++column){
			ptr[column] = in[k];
			++k;
		}
	}

	fftwf_free(in);
	fftwf_free(out);
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

int CtVolume::fftCoordToIndex(int coord, int size) const{
	if (coord < 0)return size + coord;
	return coord;
}