#include "CtVolume.h"

namespace ct {

	//constructor of data type struct "projection"

	Projection::Projection() { }

	Projection::Projection(cv::Mat image, double angle, double heightOffset) :image(image), angle(angle), heightOffset(heightOffset) {
		//empty
	}

	//============================================== PUBLIC ==============================================\\

	//constructor
	CtVolume::CtVolume() :_currentlyDisplayedImage(0), _emitSignals(false) { }

	CtVolume::CtVolume(std::string csvFile, CtVolume::FilterType filterType) : _currentlyDisplayedImage(0), _emitSignals(false) {
		sinogramFromImages(csvFile, filterType);
	}

	void CtVolume::sinogramFromImages(std::string csvFile, CtVolume::FilterType filterType) {
		//delete the contents of the sinogram
		_sinogram.clear();
		//open the csv file
		std::ifstream stream(csvFile.c_str(), std::ios::in);
		if (!stream.good()) {
			std::cerr << "Could not open CSV file - terminating" << std::endl;
			return;
		}
		//count the lines in the file
		int lineCnt = std::count(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>(), '\n') + 1;
		int imgCnt = lineCnt - 8;

		if (lineCnt > 0) {
			//resize the sinogram to the correct size
			_sinogram.resize(imgCnt);
			//go back to the beginning of the file
			stream.seekg(std::ios::beg);
			//variables for the values that shall be read
			std::stringstream strstr;
			std::string line;
			std::string path;
			std::string rotationDirection;
			double pixelSize;
			double uOffset;
			double vOffset;
			double SO;
			double SD;
			std::string file;
			double angle;
			double heightOffset;

			//manual reading of all the parameters
			std::getline(stream, path, '\t');
			stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			std::getline(stream, line);
			strstr.str(line);
			strstr.clear();
			strstr >> pixelSize;
			std::getline(stream, rotationDirection, '\t');
			stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			std::getline(stream, line);
			strstr.str(line);
			strstr.clear();
			strstr >> uOffset;
			std::getline(stream, line);
			strstr.str(line);
			strstr.clear();
			strstr >> vOffset;
			std::getline(stream, line);
			strstr.str(line);
			strstr.clear();
			strstr >> SO;
			std::getline(stream, line);
			strstr.str(line);
			strstr.clear();
			strstr >> SD;
			//leave out one line
			stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

			//now load the actual image files and their parameters
			std::cout << "Loading image files" << std::endl;
			int cnt = 0;
			int rows;
			int cols;
			while (!stream.eof()) {
				std::getline(stream, file, '\t');
				std::getline(stream, line, '\t');
				strstr.str(line);
				strstr.clear();
				strstr >> angle;
				std::getline(stream, line);
				strstr.str(line);
				strstr.clear();
				strstr >> heightOffset;
				//load the image
				_sinogram[cnt] = Projection(cv::imread(path + std::string("/") + file, CV_LOAD_IMAGE_UNCHANGED), angle, heightOffset);

				//check if everything is ok
				if (!_sinogram[cnt].image.data) {
					//if there is no image data
					_sinogram.clear();
					std::cout << "Error loading the image " << path + std::string("/") + file << ". Maybe it does not exist or permissions are missing." << std::endl;
					return;
				} else if (_sinogram[cnt].image.channels() != 1) {
					//if it has more than 1 channel
					_sinogram.clear();
					std::cout << "Error loading the image " << path + std::string("/") + file << ", it has not exactly 1 channel." << std::endl;
					return;
				} else {
					//make sure that all images have the same size
					if (cnt == 0) {
						rows = _sinogram[cnt].image.rows;
						cols = _sinogram[cnt].image.cols;
					} else {
						if (_sinogram[cnt].image.rows != rows || _sinogram[cnt].image.cols != cols) {
							//if the image has a different size than the images before stop and reverse
							_sinogram.clear();
							std::cout << "Error loading the image " << path + std::string("/") + file << ", its dimensions differ from the images before." << std::endl;
							return;
						}
					}
				}
				//convert the image to 32 bit float
				convertTo32bit(_sinogram[cnt].image);
				++cnt;
			}

			if (_sinogram.size() > 0) {
				//convert the heightOffset to a realtive value
				double sum = 0;
				for (int i = 0; i < _sinogram.size(); ++i) {
					sum += _sinogram[i].heightOffset;
				}
				sum /= (double)_sinogram.size();
				for (int i = 0; i < _sinogram.size(); ++i) {
					_sinogram[i].heightOffset -= sum;			//substract average
					_sinogram[i].heightOffset /= pixelSize;		//convert to pixels
				}
				//make sure the direction of the rotation is correct
				if (_sinogram.size() > 1) {	//only possible if there are at least 2 images
					double diff = _sinogram[1].angle - _sinogram[0].angle;
					//clockwise rotation requires rotation in positive direction and ccw rotation requires negative direction
					if ((rotationDirection == "cw" && diff < 0) || (rotationDirection == "ccw" && diff > 0)) {
						for (int i = 0; i < _sinogram.size(); ++i) {
							_sinogram[i].angle *= -1;
						}
					}
				}
				//now save the resulting size of the volume in member variables (needed for coodinate conversions later)
				_imageWidth = _sinogram[0].image.cols;
				_imageHeight = _sinogram[0].image.rows;
				//Axes: breadth = x, width = y, height = z
				_xSize = _imageWidth;
				_ySize = _imageWidth;
				_zSize = _imageHeight;
				//save the distance
				_SD = SD / pixelSize;
				_SO = SO / pixelSize;
				//save the uOffset
				_uOffset = uOffset / pixelSize;
				//now apply the filters
				imagePreprocessing(filterType);
			}
		} else {
			std::cout << "CSV file does not contain any images." << std::endl;
			return;
		}
		_minMaxCaclulated = false;
		if (_emitSignals) emit(loadingFinished(LoadStatus::SUCCESS));
	}

	cv::Mat CtVolume::sinogramImageAt(size_t index) const {
		if (index < 0 || index >= _sinogram.size()) {
			throw std::out_of_range("Index out of bounds.");
		} else {
			if (!_minMaxCaclulated) {
				_minMaxValues = getSinogramMinMaxIntensity();
				_minMaxCaclulated = true;
			}
			return normalizeImage(_sinogram[index].image, _minMaxValues.first, _minMaxValues.second);
		}
	}

	size_t CtVolume::sinogramSize() const {
		return _sinogram.size();
	}

	void CtVolume::displaySinogram(bool normalize) const {
		if (_sinogram.size() > 0) {
			if (_currentlyDisplayedImage < 0)_currentlyDisplayedImage = _sinogram.size() - 1;
			if (_currentlyDisplayedImage >= _sinogram.size())_currentlyDisplayedImage = 0;
			//normalizes all the images to the same values
			if (normalize) {
				if (!_minMaxCaclulated) {
					_minMaxValues = getSinogramMinMaxIntensity();
					_minMaxCaclulated = true;
				}
				cv::Mat normalizedImage = normalizeImage(_sinogram[_currentlyDisplayedImage].image, _minMaxValues.first, _minMaxValues.second);
				imshow("Projections", normalizedImage);
			} else {
				imshow("Projections", _sinogram[_currentlyDisplayedImage].image);
			}
			handleKeystrokes(normalize);
		} else {
			std::cout << "Could not display sinogram, it is empty." << std::endl;
		}
	}

	void CtVolume::reconstructVolume() {
		if (_sinogram.size() > 0) {
			//resize the volume to the correct size
			_volume.clear();
			_volume = std::vector<std::vector<std::vector<float>>>(_xSize, std::vector<std::vector<float>>(_ySize, std::vector<float>(_zSize, 0)));
			//mesure time
			clock_t start = clock();
			//fill the volume
			reconstructionCore();

			//now fill the corners around the cylinder with the lowest density value
			double smallestValue;
			for (int x = 0; x < _xSize; ++x) {
				for (int y = 0; y < _ySize; ++y) {
					for (int z = 0; z < _zSize; ++z) {
						if (x == 0 && y == 0 && z == 0) {
							smallestValue = _volume[x][y][z];
						} else if (_volume[x][y][z] < smallestValue) {
							smallestValue = _volume[x][y][z];
						}
					}
				}
			}

			for (int x = 0; x < _xSize; ++x) {
				for (int y = 0; y < _ySize; ++y) {
					if (sqrt(volumeToWorldX(x)*volumeToWorldX(x) + volumeToWorldY(y)*volumeToWorldY(y)) >= ((double)_xSize / 2) - 3) {
						for (int z = 0; z < _zSize; ++z) {
							_volume[x][y][z] = smallestValue;
						}
					}
				}
			}

			//mesure time
			clock_t end = clock();
			std::cout << "Volume successfully reconstructed (" << (double)(end - start) / CLOCKS_PER_SEC << "s)" << std::endl;
		} else {
			std::cout << "Volume was not reconstructed, because the sinogram seems to be empty. Please load some images first." << std::endl;
		}
	}

	void CtVolume::saveVolumeToBinaryFile(std::string filename) const {
		if (_volume.size() > 0 && _volume[0].size() > 0 && _volume[0][0].size() > 0) {

			QFile file(filename.c_str());
			if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
				std::cout << "Could not open the files. Maybe your path does not exist. No files were written." << std::endl;
				return;
			}
			QDataStream out(&file);
			out.setFloatingPointPrecision(QDataStream::SinglePrecision);
			out.setByteOrder(QDataStream::LittleEndian);
			//iterate through the volume
			for (int x = 0; x < _xSize; ++x) {
				if (_emitSignals) {
					double percentage = floor(double(x) / double(_xSize) * 100 + 0.5);
					emit(savingProgress(percentage));
				}
				for (int y = 0; y < _ySize; ++y) {
					for (int z = 0; z < _zSize; ++z) {
						//save one float of data
						out << _volume[x][y][z];
					}
				}
			}
			file.close();
			std::cout << "Volume successfully saved" << std::endl;
			if (_emitSignals) emit(savingFinished(SaveStatus::SUCCESS));
		} else {
			std::cout << "Did not save the volume, because it appears to be empty." << std::endl;
		}
	}

	void CtVolume::setEmitSignals(bool value) {
		_emitSignals = value;
	}

	//============================================== PRIVATE ==============================================\\

	std::pair<float, float> CtVolume::getSinogramMinMaxIntensity() const {
		float min = *std::min_element(_sinogram[0].image.begin<float>(), _sinogram[0].image.end<float>());
		float max = *std::max_element(_sinogram[0].image.begin<float>(), _sinogram[0].image.end<float>());
		float tmp;

		for (int i = 1; i < _sinogram.size(); ++i) {
			tmp = *std::min_element(_sinogram[i].image.begin<float>(), _sinogram[i].image.end<float>());
			if (tmp < min)min = tmp;
			tmp = *std::max_element(_sinogram[i].image.begin<float>(), _sinogram[i].image.end<float>());
			if (tmp > max)max = tmp;
		}
		return std::make_pair(min, max);
	}

	cv::Mat CtVolume::normalizeImage(cv::Mat const& image, float minValue, float maxValue) const {
		int R = image.rows;
		int C = image.cols;
		cv::Mat normalizedImage(image.rows, image.cols, CV_32F);

		const float* ptr;
		float* targPtr;
		for (int i = 0; i < R; ++i) {
			ptr = image.ptr<float>(i);
			targPtr = normalizedImage.ptr<float>(i);
			for (int j = 0; j < C; ++j) {
				targPtr[j] = (ptr[j] - minValue) / (maxValue - minValue);
			}
		}

		return normalizedImage;
	}

	void CtVolume::handleKeystrokes(bool normalize) const {
		int key = cv::waitKey(0);				//wait for a keystroke forever
		if (key == 2424832) {					//left arrow key
			--_currentlyDisplayedImage;
		} else if (key == 2555904) {				//right arrow key
			++_currentlyDisplayedImage;
		} else if (key == 27) {					//escape key
			return;
		} else if (key == -1) {					//if no key was pressed (meaning the window was closed)
			return;
		}
		if (_currentlyDisplayedImage < 0)_currentlyDisplayedImage = _sinogram.size() - 1;
		if (_currentlyDisplayedImage >= _sinogram.size())_currentlyDisplayedImage = 0;
		if (normalize) {
			cv::Mat normalizedImage = normalizeImage(_sinogram[_currentlyDisplayedImage].image, _minMaxValues.first, _minMaxValues.second);
			imshow("Projections", normalizedImage);
		} else {
			imshow("Projections", _sinogram[_currentlyDisplayedImage].image);
		}
		handleKeystrokes(normalize);
	}

	void CtVolume::imagePreprocessing(CtVolume::FilterType filterType) {
		clock_t start = clock();
		for (int i = 0; i < _sinogram.size(); ++i) {

			double percentage = floor((double)i / (double)_sinogram.size() * 100 + 0.5);
			std::cout << "\r" << "Preprocessing: " << percentage << "%";
			if (_emitSignals) emit(loadingProgress(percentage));

			applyLogScaling(_sinogram[i].image);
			applyFourierFilter(_sinogram[i].image, filterType);
			//applyWeightingFilter(_sinogram[i].image);
		}
		std::cout << std::endl;
		clock_t end = clock();
		std::cout << "Preprocessing sucessfully finished (" << (double)(end - start) / CLOCKS_PER_SEC << "s)" << std::endl;
	}

	//converts an image to 32 bit float
	//only unsigned types are allowed as input
	void CtVolume::convertTo32bit(cv::Mat& img) const {
		CV_Assert(img.depth() == CV_8U || img.depth() == CV_16U || img.depth() == CV_32F);
		if (img.depth() == CV_8U) {
			img.convertTo(img, CV_32F, 1.0 / (float)pow(2, 8));
		} else if (img.depth() == CV_16U) {
			img.convertTo(img, CV_32F, 1.0 / (float)pow(2, 16));
		}
	}

	cv::Mat CtVolume::getVolumeCrossSection() const {
		cv::Mat result(_ySize, _xSize, CV_32FC1);
		float* ptr;
		for (int row = 0; row < result.rows; ++row) {
			ptr = result.ptr<float>(row);
			for (int column = 0; column < result.cols; ++column) {
				ptr[column] = _volume[column][row][_zSize/2];
			}
		}
		return result;
	}

	//Applies a ramp filter to a 32bit float image with 1 channel; weights the center less than the borders (in horizontal direction)
	void CtVolume::applyWeightingFilter(cv::Mat& img) const {
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
		for (int i = 0; i < r; ++i) {
			ptr = img.ptr<float>(i);
			for (int j = 0; j < c; ++j) {
				factor = abs(j - centerColumn) / centerColumn;
				inverseFactor = 1 - factor;
				ptr[j] = ptr[j] * (factor * minAttenuation + inverseFactor * maxAttenuation);
			}
		}
	}

	//Applies a filter mask to a 32bit float image
	//Mind that the values after applying the filter may be greater than 1 and negative
	void CtVolume::applyHighpassFilter(cv::Mat& img) const {
		CV_Assert(img.depth() == CV_32F);
		/*cv::Mat mask = (cv::Mat_<char>(3, 3) << -1, 0, 1,
												-2, 0, 2,
												-1, 0, 1);*/
		cv::Mat mask = (cv::Mat_<char>(3, 3) << 1, 1, 1,
						1, -8, 1,
						1, 1, 1);
		cv::filter2D(img, img, img.depth(), mask, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
	}

	void CtVolume::applyFourierFilter(cv::Mat& image, CtVolume::FilterType type) const {
		CV_Assert(image.depth() == CV_32F);

		//FFT
		const int R = image.rows;
		const int C = image.cols;

		const int nyquist = (C / 2) + 1;

		float* ptr;
		fftwf_complex *out;
		out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)* nyquist);
		fftwf_plan p;

		for (int row = 0; row < R; ++row) {
			ptr = image.ptr<float>(row);
			p = fftwf_plan_dft_r2c_1d(C, ptr, out, FFTW_ESTIMATE);
			fftwf_execute(p);
			fftwf_destroy_plan(p);

			for (int i = 0; i < nyquist; ++i) {
				out[i][0] = out[i][0] / C;
				out[i][1] = out[i][1] / C;
			}

			//removing the low frequencies
			double factor;
			for (int column = 0; column < nyquist; ++column) {
				switch (type) {
					case FilterType::RAMLAK:
						factor = ramLakWindowFilter(column, nyquist);
						break;
					case FilterType::HANN:
						factor = hannWindowFilter(column, nyquist);
						break;
					case FilterType::RECTANGLE:
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

	void CtVolume::applyLogScaling(cv::Mat& image) const {
		const int R = image.rows;
		int C = image.cols;
		const double normalizationConstant = logFunction(1);
		float* ptr;
		for (int row = 0; row < R; ++row) {
			ptr = image.ptr<float>(row);
			for (int cols = 0; cols < C; ++cols) {
				ptr[cols] = 1 - logFunction(ptr[cols]) / normalizationConstant;
			}
		}
	}

	double CtVolume::logFunction(double x) const {
		const double compressionFactor = 0.005; //must be greater than 0; the closer to 0, the stronger the compression
		return std::log(x + compressionFactor) - std::log(compressionFactor);
	}

	double CtVolume::ramLakWindowFilter(double n, double N) const {
		return (double)n / (double)N;
	}

	double CtVolume::hannWindowFilter(double n, double N) const {
		return ramLakWindowFilter(n, N) * 0.5*(1 + cos((2 * M_PI * (double)n) / ((double)N * 2)));
	}

	double CtVolume::rectangleWindowFilter(double n, double N) const {
		double cutoffRatio = 0.09;		//defining the width of the filter rectangle, should be in interval [0,1]
		if (n <= cutoffRatio*N) {
			return 0;
		}
		return 1;
	}

	//This filter is not used, code could theoretically be removed
	void CtVolume::applyFourierHighpassFilter2D(cv::Mat& image) const {
		CV_Assert(image.depth() == CV_32F);

		//FFT
		const int R = image.rows;
		const int C = image.cols;

		std::vector<std::vector<std::complex<double>>> result(R, std::vector<std::complex<double>>(C));
		float *in;
		in = (float*)fftwf_malloc(sizeof(float)* R * C);
		int k = 0;
		float* ptr;
		for (int row = 0; row < R; ++row) {
			ptr = image.ptr<float>(row);
			for (int column = 0; column < C; ++column) {
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
		for (int row = 0; row < R; ++row) {
			for (int column = 0; column < nyquist; ++column) {
				out[k][0] = out[k][0] / (R*C);
				out[k][1] = out[k][1] / (R*C);
				++k;
			}
		}

		//removing the low frequencies

		double cutoffRatio = 0.1;
		int uCutoff = (cutoffRatio / 2.0)*C;
		int vCutoff = (cutoffRatio / 2.0)*R;
		for (int row = -vCutoff; row <= vCutoff; ++row) {
			for (int column = 0; column <= uCutoff; ++column) {
				out[fftCoordToIndex(row, R)*nyquist + column][0] = 0;
				out[fftCoordToIndex(row, R)*nyquist + column][1] = 0;
			}
		}

		//reverse FFT

		p = fftwf_plan_dft_c2r_2d(R, C, out, in, FFTW_ESTIMATE);
		fftwf_execute(p);
		fftwf_destroy_plan(p);
		k = 0;
		for (int row = 0; row < R; ++row) {
			ptr = image.ptr<float>(row);
			for (int column = 0; column < C; ++column) {
				ptr[column] = in[k];
				++k;
			}
		}

		fftwf_free(in);
		fftwf_free(out);
	}

	void CtVolume::reconstructionCore() {
		double imageLowerBoundU = matToImageU(0);
		double imageUpperBoundU = matToImageU(_imageWidth - 1);
		double imageLowerBoundV = matToImageV(0);
		double imageUpperBoundV = matToImageV(_imageHeight - 1);

		int volumeLowerBoundX = volumeToWorldX(0);
		int volumeUpperBoundX = volumeToWorldX(_xSize);
		int volumeLowerBoundY = volumeToWorldY(0);
		int volumeUpperBoundY = volumeToWorldY(_ySize);
		int volumeLowerBoundZ = volumeToWorldZ(0);
		int volumeUpperBoundZ = volumeToWorldZ(_zSize);


#pragma omp parallel for schedule(dynamic)
		for (int projection = 0; projection < _sinogram.size(); ++projection) {

			//output percentage
			if (omp_get_thread_num() == 0) {
				double percentage = floor((double)projection / (double)_sinogram.size() * 100 + 0.5);
				std::cout << "\r" << "Backprojecting: " << percentage << "%";
				if (_emitSignals) emit(reconstructionProgress(percentage, getVolumeCrossSection()));
			}
			double beta_rad = (_sinogram[projection].angle / 180.0) * M_PI;
			double sine = sin(beta_rad);
			double cosine = cos(beta_rad);
			//copy some member variables to local variables, performance is better this way
			cv::Mat image = _sinogram[projection].image;
			double heightOffset = _sinogram[projection].heightOffset;
			double uOffset = _uOffset;
			double SD = _SD;

			for (int x = volumeLowerBoundX; x < volumeUpperBoundX; ++x) {
				for (int y = volumeLowerBoundY; y < volumeUpperBoundY; ++y) {
					if (sqrt((double)x*(double)x + (double)y*(double)y) < ((double)_xSize / 2.0) - 3) {
						//if the voxel is inside the reconstructable cylinder
						for (int z = volumeLowerBoundZ; z < volumeUpperBoundZ; ++z) {

							double t = (-1)*double(x)*sine + double(y)*cosine;
							double s = double(x)*cosine + double(y)*sine;
							double u = (t*SD) / (SD - s);
							double v = ((double(z) - heightOffset)*SD) / (SD - s);
							//correct the u-offset
							u += uOffset;

							//check if it's inside the image (before the coordinate transformation)
							if (u >= imageLowerBoundU && u <= imageUpperBoundU && v >= imageLowerBoundV && v <= imageUpperBoundV) {

								double weight = W(SD, u, v);

								u = imageToMatU(u);
								v = imageToMatV(v);

								//get the 4 surrounding pixels for the bilinear interpolation (note: u and v are always positive)
								int u0 = u;
								int u1 = u0 + 1;
								int v0 = v;
								int v1 = v0 + 1;

								//check if all the pixels are inside the image (after the coordinate transformation) (probably not necessary)
								//if (u0 < _imageWidth && u0 >= 0 && u1 < _imageWidth && u1 >= 0 && v0 < _imageHeight && v0 >= 0 && v1 < _imageHeight && v1 >= 0) {

								float* row = image.ptr<float>(v0);
								float u0v0 = row[u0];
								float u1v0 = row[u1];
								row = image.ptr<float>(v1);
								float u0v1 = row[u0];
								float u1v1 = row[u1];
								_volume[worldToVolumeX(x)][worldToVolumeY(y)][worldToVolumeZ(z)] += weight * bilinearInterpolation(u - double(u0), v - double(v0), u0v0, u1v0, u0v1, u1v1);
							}
						}
					}
				}
			}
		}
		std::cout << std::endl;
		if (_emitSignals) emit(reconstructionFinished(ReconstructStatus::SUCCESS, getVolumeCrossSection()));
	}

	inline float CtVolume::bilinearInterpolation(double u, double v, float u0v0, float u1v0, float u0v1, float u1v1) const {
		//the two interpolations on the u axis
		double v0 = (1 - u)*u0v0 + u*u1v0;
		double v1 = (1 - u)*u0v1 + u*u1v1;
		//interpolation on the v axis between the two u-interpolated values
		return (1 - v)*v0 + v*v1;
	}

	inline double CtVolume::W(double D, double u, double v) {
		return D / sqrt(D*D + u*u + v*v);
	}

	inline double CtVolume::worldToVolumeX(double xCoord) const {
		return xCoord + ((double)_xSize / 2.0);
	}

	inline double CtVolume::worldToVolumeY(double yCoord) const {
		return yCoord + ((double)_ySize / 2.0);
	}

	inline double CtVolume::worldToVolumeZ(double zCoord) const {
		return zCoord + ((double)_zSize / 2.0);
	}

	inline double CtVolume::volumeToWorldX(double xCoord) const {
		return xCoord - ((double)_xSize / 2.0);
	}

	inline double CtVolume::volumeToWorldY(double yCoord) const {
		return yCoord - ((double)_ySize / 2.0);
	}

	inline double CtVolume::volumeToWorldZ(double zCoord) const {
		return zCoord - ((double)_zSize / 2.0);
	}

	inline double CtVolume::imageToMatU(double uCoord)const {
		return uCoord + ((double)_imageWidth / 2.0);
	}

	inline double CtVolume::imageToMatV(double vCoord)const {
		return vCoord + ((double)_imageHeight / 2.0);
	}

	inline double CtVolume::matToImageU(double uCoord)const {
		return uCoord - ((double)_imageWidth / 2.0);
	}

	inline double CtVolume::matToImageV(double vCoord)const {
		return vCoord - ((double)_imageHeight / 2.0);
	}

	int CtVolume::fftCoordToIndex(int coord, int size) const {
		if (coord < 0)return size + coord;
		return coord;
	}

}