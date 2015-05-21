#include "CtVolume.h"

namespace ct {

	//constructor of data type struct "projection"

	Projection::Projection() { }

	Projection::Projection(cv::Mat image, double angle, double heightOffset) :image(image), angle(angle), heightOffset(heightOffset) {
		//empty
	}

	//============================================== PUBLIC ==============================================\\

	//constructor
	CtVolume::CtVolume()
		:_currentlyDisplayedImage(0),
		_emitSignals(false),
		_crossSectionIndex(0),
		_stop(false),
		_xSize(0),
		_ySize(0),
		_zSize(0),
		_imageWidth(0),
		_imageHeight(0),
		_uOffset(0),
		_vOffset(0),
		_pixelSize(0),
		_SO(0),
		_SD(0),
		_xFrom(0),
		_xTo(0),
		_yFrom(0),
		_yTo(0),
		_zFrom(0),
		_zTo(0),
		_xMax(0),
		_yMax(0),
		_zMax(0),
		_xFrom_float(0),
		_xTo_float(1),
		_yFrom_float(0),
		_yTo_float(1),
		_zFrom_float(0),
		_zTo_float(1) { }

	CtVolume::CtVolume(std::string csvFile, FilterType filterType)
		: _currentlyDisplayedImage(0),
		_crossSectionIndex(0),
		_stop(false),
		_emitSignals(false),
		_xSize(0),
		_ySize(0),
		_zSize(0),
		_imageWidth(0),
		_imageHeight(0),
		_uOffset(0),
		_vOffset(0),
		_pixelSize(0),
		_SO(0),
		_SD(0),
		_xFrom(0),
		_xTo(0),
		_yFrom(0),
		_yTo(0),
		_zFrom(0),
		_zTo(0),
		_xMax(0),
		_yMax(0),
		_zMax(0),
		_xFrom_float(0),
		_xTo_float(1),
		_yFrom_float(0),
		_yTo_float(1),
		_zFrom_float(0),
		_zTo_float(1) {
		sinogramFromImages(csvFile, filterType);
	}

	void CtVolume::sinogramFromImages(std::string csvFile, FilterType filterType) {
		_stop = false;
		_volume.clear();
		//delete the contents of the sinogram
		_sinogram.clear();
		//open the csv file
		std::ifstream stream(csvFile.c_str(), std::ios::in);
		if (!stream.good()) {
			std::cerr << "Could not open CSV file - terminating" << std::endl;
			if (_emitSignals) emit(loadingFinished(CompletionStatus::error("Could not open the config file.")));
			return;
		}
		//count the lines in the file
		int lineCnt = std::count(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>(), '\n') + 1;
		int imgCnt = lineCnt - 8;

		if (imgCnt <= 0) {
			std::cout << "CSV file does not contain any images." << std::endl;
			if (_emitSignals) emit(loadingFinished(CompletionStatus::error("Apparently the config file does not contain any images.")));
			return;
		}else{
			//resize the sinogram to the correct size
			_sinogram.reserve(imgCnt);

			//go back to the beginning of the file
			stream.seekg(std::ios::beg);
			
			//read the parameter section of the csv file
			std::string path;
			std::string rotationDirection;
			readParameters(stream, path, rotationDirection);

			//the image path might be relative
			path = glueRelativePath(csvFile, path);

			//read the images from the csv file
			if (!readImages(stream, path)) return;
			if (_stop) {
				_sinogram.clear();
				std::cout << "User interrupted. Stopping.";
				if (_emitSignals) emit(loadingFinished(CompletionStatus::interrupted()));
				return;
			}

			if (_sinogram.size() > 0) {
				//make the height offset values realtive
				makeHeightOffsetRelative();
				//make sure the rotation direction is correct
				correctAngleDirection(rotationDirection);
				//now save the resulting size of the volume in member variables (needed for coodinate conversions later)
				_imageWidth = _sinogram[0].image.cols;
				_imageHeight = _sinogram[0].image.rows;
				//Axes: breadth = x, width = y, height = z
				_xSize = _imageWidth;
				_ySize = _imageWidth;
				_zSize = _imageHeight;
				updateBoundaries();
				_crossSectionIndex = _zMax / 2;
				//now apply the filters
				imagePreprocessing(filterType);
				if (_stop) {
					_sinogram.clear();
					std::cout << "User interrupted. Stopping.";
					if (_emitSignals) emit(loadingFinished(CompletionStatus::interrupted()));
					return;
				}
			}
		}
		_minMaxCaclulated = false;
		if (_emitSignals) emit(loadingFinished());
	}

	Projection CtVolume::getProjectionAt(size_t index) const {
		if (index < 0 || index >= _sinogram.size()) {
			throw std::out_of_range("Index out of bounds.");
		} else {
			if (!_minMaxCaclulated) {
				_minMaxValues = getSinogramMinMaxIntensity();
				_minMaxCaclulated = true;
			}
			Projection local(_sinogram[index]);
			local.image = normalizeImage(local.image, _minMaxValues.first, _minMaxValues.second);
			return local;
		}
	}

	size_t CtVolume::getSinogramSize() const {
		if (_sinogram.size() > 0) {
			return _sinogram.size();
		}
		return 0;
	}

	size_t CtVolume::getImageWidth() const {
		return _imageWidth;
	}

	size_t CtVolume::getImageHeight() const {
		return _imageHeight;
	}

	size_t CtVolume::getXSize() const {
		return _xMax;
	}

	size_t CtVolume::getYSize() const {
		return _yMax;
	}

	size_t CtVolume::getZSize() const {
		return _zMax;
	}

	double CtVolume::getUOffset() const {
		return _uOffset;
	}

	double CtVolume::getVOffset() const {
		return _vOffset;
	}

	double CtVolume::getPixelSize() const {
		return _pixelSize;
	}

	double CtVolume::getSO() const {
		return _SO;
	}

	double CtVolume::getSD() const {
		return _SD;
	}

	cv::Mat CtVolume::getVolumeCrossSection(size_t zCoord) const {
		if (zCoord >= 0 && zCoord < _zMax) {
			if (_volume.size() > 0 && _volume[0].size() > 0 && _volume[0][0].size() > 0) {
				cv::Mat result(_volume[0].size(), _volume.size(), CV_32FC1);
				float* ptr;
#pragma omp parallel for
				for (int row = 0; row < result.rows; ++row) {
					ptr = result.ptr<float>(row);
					for (int column = 0; column < result.cols; ++column) {
						ptr[column] = _volume[column][row][zCoord];
					}
				}
				return result;
			}
			return cv::Mat();
		} else {
			throw std::out_of_range("Index out of bounds.");
		}
	}

	void CtVolume::setCrossSectionIndex(size_t zCoord) {
		if (zCoord >= 0 && zCoord < _zMax) {
			_crossSectionIndex = zCoord;
		}
	}

	size_t CtVolume::getCrossSectionIndex() const {
		return _crossSectionIndex;
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

	void CtVolume::setVolumeBounds(double xFrom, double xTo, double yFrom, double yTo, double zFrom, double zTo) {
		_xFrom_float = std::max(0.0, std::min(1.0, xFrom));
		_xTo_float = std::max(_xFrom_float, std::min(1.0, xTo));
		_yFrom_float = std::max(0.0, std::min(1.0, yFrom));
		_yTo_float = std::max(_xFrom_float, std::min(1.0, yTo));
		_zFrom_float = std::max(0.0, std::min(1.0, zFrom));
		_zTo_float = std::max(_xFrom_float, std::min(1.0, zTo));
		if (_sinogram.size() > 0) updateBoundaries();
	}

	void CtVolume::reconstructVolume() {
		_stop = false;
		if (_sinogram.size() > 0) {
			//resize the volume to the correct size
			_volume = std::vector<std::vector<std::vector<float>>>(_xMax, std::vector<std::vector<float>>(_yMax, std::vector<float>(_zMax, 0)));
			//mesure time
			clock_t start = clock();
			//fill the volume
			reconstructionCore();
			if (_stop) {
				_volume.clear();
				std::cout << "User interrupted. Stopping." << std::endl;
				if (_emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::interrupted()));
				return;
			}

			//now fill the corners around the cylinder with the lowest density value
			if (_xMax > 0 && _yMax > 0 && _zMax > 0) {
				double smallestValue;
				smallestValue = _volume[0][0][0];
				for (int x = 0; x < _xMax; ++x) {
					for (int y = 0; y < _yMax; ++y) {
						for (int z = 0; z < _zMax; ++z) {
							if (_volume[x][y][z] < smallestValue) {
								smallestValue = _volume[x][y][z];
							}
						}
					}
				}

				for (int x = 0; x < _xMax; ++x) {
					for (int y = 0; y < _yMax; ++y) {
						if (sqrt(volumeToWorldX(x)*volumeToWorldX(x) + volumeToWorldY(y)*volumeToWorldY(y)) >= ((double)_xSize / 2) - 3) {
							for (int z = 0; z < _zMax; ++z) {
								_volume[x][y][z] = smallestValue;
							}
						}
					}
				}
			}

			//mesure time
			clock_t end = clock();
			std::cout << "Volume successfully reconstructed (" << (double)(end - start) / CLOCKS_PER_SEC << "s)" << std::endl;
			if (_emitSignals) emit(reconstructionFinished(getVolumeCrossSection(_crossSectionIndex)));
		} else {
			std::cout << "Volume was not reconstructed, because the sinogram seems to be empty. Please load some images first." << std::endl;
			if (_emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::error("Volume was not reconstructed, because the sinogram seems to be empty. Please load some images first.")));
		}
	}

	void CtVolume::saveVolumeToBinaryFile(std::string filename) const {
		_stop = false;
		if (_volume.size() > 0 && _volume[0].size() > 0 && _volume[0][0].size() > 0) {

			QFile file(filename.c_str());
			if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
				std::cout << "Could not open the file. Maybe your path does not exist. No files were written." << std::endl;
				if (_emitSignals) emit(savingFinished(CompletionStatus::error("Could not open the file. Maybe your path does not exist. No files were written.")));
				return;
			}
			QDataStream out(&file);
			out.setFloatingPointPrecision(QDataStream::SinglePrecision);
			out.setByteOrder(QDataStream::LittleEndian);
			//iterate through the volume
			for (int x = 0; x < _xMax; ++x) {
				if (_stop) break;
				if (_emitSignals) {
					double percentage = floor(double(x) / double(_volume.size()) * 100 + 0.5);
					emit(savingProgress(percentage));
				}
				for (int y = 0; y < _yMax; ++y) {
					for (int z = 0; z < _zMax; ++z) {
						//save one float of data
						out << _volume[x][y][z];
					}
				}
			}
			file.close();
			if (_stop) {
				std::cout << "User interrupted. Stopping." << std::endl;
				if (_emitSignals) emit(savingFinished(CompletionStatus::interrupted()));
				return;
			}
			std::cout << "Volume successfully saved." << std::endl;
			if (_emitSignals) emit(savingFinished());
		} else {
			std::cout << "Did not save the volume, because it appears to be empty." << std::endl;
			if (_emitSignals) emit(savingFinished(CompletionStatus::error("Did not save the volume, because it appears to be empty.")));
		}
	}

	void CtVolume::stop() {
		_stop = true;
	}

	void CtVolume::setEmitSignals(bool value) {
		_emitSignals = value;
	}

	//============================================== PRIVATE ==============================================\\

	void CtVolume::readParameters(std::ifstream& stream, std::string& path, std::string& rotationDirection) {
		//variables for the values that shall be read
		std::string line;
		std::stringstream lineStream;
		std::string field;

		//manual reading of all the parameters
		std::getline(stream, path, '\t');
		stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		std::getline(stream, line);
		lineStream.str(line);
		lineStream.clear();
		lineStream >> _pixelSize;
		std::getline(stream, line);
		lineStream.str(line);
		lineStream.clear();
		std::getline(lineStream, rotationDirection, '\t');
		std::cout << rotationDirection << std::endl;
		std::getline(stream, line);
		lineStream.str(line);
		lineStream.clear();
		lineStream >> _uOffset;
		std::getline(stream, line);
		lineStream.str(line);
		lineStream.clear();
		lineStream >> _vOffset;
		std::getline(stream, line);
		lineStream.str(line);
		lineStream.clear();
		lineStream >> _SO;
		std::getline(stream, line);
		lineStream.str(line);
		lineStream.clear();
		lineStream >> _SD;
		//leave out one line
		stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

		//convert the distance
		_SD /= _pixelSize;
		_SO /= _pixelSize;
		//convert uOffset and vOffset
		_uOffset /= _pixelSize;
		_vOffset /= _pixelSize;
	}

	std::string CtVolume::glueRelativePath(std::string const& basePath, std::string const& potentialRelativePath) {
		//handle relative paths
		std::string resultPath;
		if (potentialRelativePath.size() > 0 && potentialRelativePath.at(0) == '.') {
			size_t pos = basePath.find_last_of("/\\");
			if (pos != std::string::npos) {
				std::string folder = basePath.substr(0, pos + 1);
				resultPath = folder + potentialRelativePath;
			}
		} else {
			resultPath = potentialRelativePath;
		}
		if (resultPath.size() > 0 && *(resultPath.end() - 1) != '/' && *(resultPath.end() - 1) != '\\') {
			resultPath = resultPath + std::string("/");
		}
		return resultPath;
	}

	bool CtVolume::readImages(std::ifstream& csvStream, std::string path) {
		//now load the actual image files and their parameters
		std::cout << "Loading image files" << std::endl;
		std::stringstream lineStream;
		std::string line;
		std::string field;
		std::string file;
		double angle;
		double heightOffset;
		int cnt = 0;
		int rows;
		int cols;
		while (std::getline(csvStream, line) && !_stop) {
			lineStream.str(line);
			lineStream.clear();
			std::getline(lineStream, file, '\t');
			std::getline(lineStream, field, '\t');
			angle = std::stod(field);
			std::getline(lineStream, field);
			heightOffset = std::stod(field);
			//load the image
			_sinogram.push_back(Projection(cv::imread(path + file, CV_LOAD_IMAGE_UNCHANGED), angle, heightOffset));

			//check if everything is ok
			if (!_sinogram[cnt].image.data) {
				//if there is no image data
				_sinogram.clear();
				std::string msg = "Error loading the image \"" + path + file + "\" (line " + std::to_string(cnt + 9) + "). Maybe it does not exist or permissions are missing.";
				std::cout << msg << std::endl;
				if (_emitSignals) emit(loadingFinished(CompletionStatus::error(msg.c_str())));
				return false;
			} else if (_sinogram[cnt].image.channels() != 1) {
				//if it has more than 1 channel
				_sinogram.clear();
				std::string msg = "Error loading the image \"" + path + file + "\", it has not exactly 1 channel.";
				std::cout << msg << std::endl;
				if (_emitSignals) emit(loadingFinished(CompletionStatus::error(msg.c_str())));
				return false;
			} else {
				//make sure that all images have the same size
				if (cnt == 0) {
					rows = _sinogram[cnt].image.rows;
					cols = _sinogram[cnt].image.cols;
				} else {
					if (_sinogram[cnt].image.rows != rows || _sinogram[cnt].image.cols != cols) {
						//if the image has a different size than the images before stop and reverse
						_sinogram.clear();
						std::string msg = "Error loading the image \"" + file + "\", its dimensions differ from the images before.";
						std::cout << msg << std::endl;
						if (_emitSignals) emit(loadingFinished(CompletionStatus::error(msg.c_str())));
						return false;
					}
				}
			}
			//convert the image to 32 bit float
			convertTo32bit(_sinogram[cnt].image);
			++cnt;
		}
		return true;
	}

	void CtVolume::makeHeightOffsetRelative() {
		//convert the heightOffset to a realtive value
		double sum = 0;
		for (int i = 0; i < _sinogram.size(); ++i) {
			sum += _sinogram[i].heightOffset;
		}
		sum /= (double)_sinogram.size();
		for (int i = 0; i < _sinogram.size(); ++i) {
			_sinogram[i].heightOffset -= sum;			//substract average
			_sinogram[i].heightOffset /= _pixelSize;		//convert to pixels
		}
	}

	void CtVolume::correctAngleDirection(std::string rotationDirection) {
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
	}

	std::pair<float, float> CtVolume::getSinogramMinMaxIntensity() const {
		double min = 0;
		double max = 0;
		cv::minMaxLoc(_sinogram[0].image, &min, &max);
		for (int i = 1; i < _sinogram.size(); ++i) {
			double lMin, lMax;
			cv::minMaxLoc(_sinogram[i].image, &lMin, &lMax);
			if (lMin < min)min = lMin;
			if (lMax > max)max = lMax;
		}
		return std::make_pair(float(min), float(max));
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

	void CtVolume::imagePreprocessing(FilterType filterType) {
		clock_t start = clock();
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < _sinogram.size(); ++i) {
			if (!_stop) {
				double percentage = floor((double)i / (double)_sinogram.size() * 100 + 0.5);
				if (omp_get_thread_num() == 0) std::cout << "\r" << "Preprocessing: " << percentage << "%";
				if (_emitSignals) emit(loadingProgress(percentage));

				applyLogScaling(_sinogram[i].image);
				applyFourierFilterOpenCV(_sinogram[i].image, filterType);
				applyFeldkampWeight(_sinogram[i].image);
			}
		}
		std::cout << std::endl;
		clock_t end = clock();
		std::cout << "Preprocessing sucessfully finished (" << (double)(end - start) / CLOCKS_PER_SEC << "s)" << std::endl;
	}

	//converts an image to 32 bit float
	//only unsigned types are allowed as input
	void CtVolume::convertTo32bit(cv::Mat& img) {
		CV_Assert(img.depth() == CV_8U || img.depth() == CV_16U || img.depth() == CV_32F);
		if (img.depth() == CV_8U) {
			img.convertTo(img, CV_32F, 1.0 / (float)pow(2, 8));
		} else if (img.depth() == CV_16U) {
			img.convertTo(img, CV_32F, 1.0 / (float)pow(2, 16));
		}
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

	void CtVolume::applyFeldkampWeight(cv::Mat& image) const {
		CV_Assert(image.channels() == 1);
		CV_Assert(image.depth() == CV_32F);

		float* ptr;
		for (int r = 0; r < image.rows; ++r) {
			ptr = image.ptr<float>(r);
			for (int c = 0; c < image.cols; ++c) {
				ptr[c] = ptr[c] * W(_SD, matToImageU(c), matToImageV(r));
			}
		}
	}

	void CtVolume::applyFourierFilter(cv::Mat& image, FilterType type) {
		CV_Assert(image.depth() == CV_32F);

		//FFT
		const int R = image.rows;
		const int C = image.cols;

		const int nyquist = (C / 2) + 1;

		float* ptr = image.ptr<float>(0);
		fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)* nyquist);

		fftwf_plan plan = fftwf_plan_dft_r2c_1d(C, ptr, out, FFTW_ESTIMATE);
		fftwf_plan planInverse = fftwf_plan_dft_c2r_1d(C, out, ptr, FFTW_ESTIMATE);

		for (int row = 0; row < R; ++row) {
			ptr = image.ptr<float>(row);
			fftwf_execute_dft_r2c(plan, ptr, out);
			
			//scaling not necessary
			//for (int i = 0; i < nyquist; ++i) {
			//	out[i][0] = out[i][0] / C;
			//	out[i][1] = out[i][1] / C;
			//}

			//removing the low frequencies
			double factor;
			for (int column = 0; column < nyquist; ++column) {
				switch (type) {
					case FilterType::RAMLAK:
						factor = ramLakWindowFilter(column, nyquist);
						break;
					case FilterType::SHEPP_LOGAN:
						factor = sheppLoganWindowFilter(column, nyquist);
						break;
					case FilterType::HANN:
						factor = hannWindowFilter(column, nyquist);
						break;
					default:
						factor = ramLakWindowFilter(column, nyquist);
						break;
				}
				out[column][0] *= factor;
				out[column][1] *= factor;
			}

			//inverse
			fftwf_execute_dft_c2r(planInverse, out, ptr);
		}

		fftwf_destroy_plan(plan);
		fftwf_destroy_plan(planInverse);
		fftwf_free(out);
	}

	void CtVolume::applyFourierFilterOpenCV(cv::Mat& image, FilterType type) {
		//cv::Mat m = (cv::Mat_<float>(1, 8) << 3, 8, 2, 10, 5, 20, 9, 2);
		//cv::Mat output;
		//cv::dft(m, output, cv::DFT_COMPLEX_OUTPUT);
		//std::cout << cv::format(output, "python") << std::endl;
		//system("pause");

		cv::Mat freq;
		cv::dft(image, freq, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);
		unsigned int nyquist = (freq.cols / 2) + 1;
		cv::Vec2f* ptr;
		for (int row = 0; row < freq.rows; ++row) {
			ptr = freq.ptr<cv::Vec2f>(row);
			for (int column = 0; column < nyquist; ++column) {
				switch (type) {
					case FilterType::RAMLAK:
						ptr[column] *= ramLakWindowFilter(column, nyquist);
						break;
					case FilterType::SHEPP_LOGAN:
						ptr[column] *= sheppLoganWindowFilter(column, nyquist);
						break;
					case FilterType::HANN:
						ptr[column] *= hannWindowFilter(column, nyquist);
						break;
				}
			}
		}
		cv::idft(freq, image, cv::DFT_ROWS | cv::DFT_REAL_OUTPUT);
	}

	void CtVolume::applyLogScaling(cv::Mat& image) {
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

	double CtVolume::logFunction(double x) {
		const double compressionFactor = 0.005; //must be greater than 0; the closer to 0, the stronger the compression
		return std::log(x + compressionFactor) - std::log(compressionFactor);
	}

	double CtVolume::ramLakWindowFilter(double n, double N) {
		return double(n) / double(N);
	}

	double CtVolume::sheppLoganWindowFilter(double n, double N) {
		if (n == 0) {
			return 0;
		} else {
			double rl = ramLakWindowFilter(n, N);
			return (rl)* (sin(rl*0.5*M_PI)) / (rl*0.5*M_PI);
		}

	}

	double CtVolume::hannWindowFilter(double n, double N) {
		return ramLakWindowFilter(n, N) * 0.5*(1 + cos((2 * M_PI * double(n)) / (double(N) * 2)));
	}

	//This filter is not used, code could theoretically be removed
	void CtVolume::applyFourierHighpassFilter2D(cv::Mat& image) {
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
		//inversed because of inversed v axis in mat/image coordinate system
		double imageLowerBoundV = matToImageV(_imageHeight - 1);
		double imageUpperBoundV = matToImageV(0);

		double volumeLowerBoundY = volumeToWorldY(0);
		double volumeUpperBoundY = volumeToWorldY(_yMax);
		double volumeLowerBoundZ = volumeToWorldZ(0);
		double volumeUpperBoundZ = volumeToWorldZ(_zMax);


		for (int projection = 0; projection < _sinogram.size(); ++projection) {
			if (_stop) {
				std::cout << "User interrupted. Stopping." << std::endl;
				if (_emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::interrupted()));
				return;
			}
			//output percentage
			double percentage = floor((double)projection / (double)_sinogram.size() * 100 + 0.5);
			std::cout << "\r" << "Backprojecting: " << percentage << "%";
			if (_emitSignals) emit(reconstructionProgress(percentage, getVolumeCrossSection(_crossSectionIndex)));
			double beta_rad = (_sinogram[projection].angle / 180.0) * M_PI;
			double sine = sin(beta_rad);
			double cosine = cos(beta_rad);
			//copy some member variables to local variables, performance is better this way
			cv::Mat image = _sinogram[projection].image;
			double heightOffset = _sinogram[projection].heightOffset;
			double uOffset = _uOffset;
			double SD = _SD;

#pragma omp parallel for schedule(dynamic)
			for (int xIndex = 0; xIndex < _xMax; ++xIndex) {
				double x = volumeToWorldX(xIndex);
				for (double y = volumeLowerBoundY; y < volumeUpperBoundY; ++y) {
					if (sqrt(x*x + y*y) < (_xSize / 2.0) - 3) {
						//if the voxel is inside the reconstructable cylinder
						for (double z = volumeLowerBoundZ; z < volumeUpperBoundZ; ++z) {

							double t = (-1)*x*sine + y*cosine;
							//correct the u-offset
							t += uOffset;
							double s = x*cosine + y*sine;
							double u = (t*SD) / (SD - s);
							double v = ((z + heightOffset)*SD) / (SD - s);

							//check if it's inside the image (before the coordinate transformation)
							if (u >= imageLowerBoundU && u <= imageUpperBoundU && v >= imageLowerBoundV && v <= imageUpperBoundV) {

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
								_volume[worldToVolumeX(x)][worldToVolumeY(y)][worldToVolumeZ(z)] += bilinearInterpolation(u - double(u0), v - double(v0), u0v0, u1v0, u0v1, u1v1);
							}
						}
					}
				}
			}
		}
		std::cout << std::endl;
	}

	inline float CtVolume::bilinearInterpolation(double u, double v, float u0v0, float u1v0, float u0v1, float u1v1) {
		//the two interpolations on the u axis
		double v0 = (1.0 - u)*u0v0 + u*u1v0;
		double v1 = (1.0 - u)*u0v1 + u*u1v1;
		//interpolation on the v axis between the two u-interpolated values
		return (1.0 - v)*v0 + v*v1;
	}

	inline double CtVolume::W(double D, double u, double v) {
		return D / sqrt(D*D + u*u + v*v);
	}

	void CtVolume::updateBoundaries() {
		//calcualte bounds (+0.5 for correct rouding)
		_xFrom = _xFrom_float * _xSize + 0.5;
		_xTo = _xTo_float * _xSize + 0.5;
		_yFrom = _yFrom_float * _ySize + 0.5;
		_yTo = _yTo_float * _ySize + 0.5;
		_zFrom = _zFrom_float * _zSize + 0.5;
		_zTo = _zTo_float * _zSize + 0.5;
		_xMax = _xTo - _xFrom;
		_yMax = _yTo - _yFrom;
		_zMax = _zTo - _zFrom;
	}

	inline double CtVolume::worldToVolumeX(double xCoord) const {
		return xCoord + ((double)_xSize / 2.0) - double(_xFrom);
	}

	inline double CtVolume::worldToVolumeY(double yCoord) const {
		return yCoord + ((double)_ySize / 2.0) - double(_yFrom);
	}

	inline double CtVolume::worldToVolumeZ(double zCoord) const {
		return zCoord + ((double)_zSize / 2.0) - double(_zFrom);
	}

	inline double CtVolume::volumeToWorldX(double xCoord) const {
		return xCoord - (double(_xSize) / 2.0) + double(_xFrom);
	}

	inline double CtVolume::volumeToWorldY(double yCoord) const {
		return yCoord - (double(_ySize) / 2.0) + double(_yFrom);
	}

	inline double CtVolume::volumeToWorldZ(double zCoord) const {
		return zCoord - (double(_zSize) / 2.0) + double(_zFrom);
	}

	inline double CtVolume::imageToMatU(double uCoord)const {
		return uCoord + ((double)_imageWidth / 2.0);
	}

	inline double CtVolume::imageToMatV(double vCoord)const {
		//factor -1 because of different z-axis direction
		return ((-1)*vCoord + ((double)_imageHeight / 2.0));
	}

	inline double CtVolume::matToImageU(double uCoord)const {
		return uCoord - ((double)_imageWidth / 2.0);
	}

	inline double CtVolume::matToImageV(double vCoord)const {
		//factor -1 because of different z-axis direction
		return (-1)*(vCoord - ((double)_imageHeight / 2.0));
	}

	int CtVolume::fftCoordToIndex(int coord, int size) {
		if (coord < 0)return size + coord;
		return coord;
	}

}