#include "CtVolume.h"

namespace ct {

	//data type projection visible to the outside

	Projection::Projection() { }

	Projection::Projection(cv::Mat image, double angle, double heightOffset) : image(image), angle(angle), heightOffset(heightOffset) { }


	//internal data type projection

	CtVolume::Projection::Projection() { }

	CtVolume::Projection::Projection(std::string imagePath, double angle, double heightOffset) : imagePath(imagePath), angle(angle), heightOffset(heightOffset) {
		//empty
	}

	cv::Mat CtVolume::Projection::getImage() const {
		return cv::imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
	}

	ct::Projection CtVolume::Projection::getPublicProjection() const {
		return ct::Projection(getImage(), angle, heightOffset);
	}

	//============================================== PUBLIC ==============================================\\

	//constructor
	CtVolume::CtVolume(std::string csvFile) {
		sinogramFromImages(csvFile);
	}

	void CtVolume::sinogramFromImages(std::string csvFile) {
		std::lock_guard<std::mutex> lock(_exclusiveFunctionsMutex);
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
			if (!readImages(stream, path, imgCnt)) return;
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
				//Axes: breadth = x, width = y, height = z
				_xSize = _imageWidth;
				_ySize = _imageWidth;
				_zSize = _imageHeight;
				updateBoundaries();
				switch (_crossSectionAxis) {
					case Axis::X:
						_crossSectionIndex = _xMax / 2;
						break;
					case Axis::Y:
						_crossSectionIndex = _yMax / 2;
						break;
					case Axis::Z:
						_crossSectionIndex = _zMax / 2;
						break;
				}
				if (_stop) {
					_sinogram.clear();
					std::cout << "User interrupted. Stopping.";
					if (_emitSignals) emit(loadingFinished(CompletionStatus::interrupted()));
					return;
				}
			}
		}
		if (_emitSignals) emit(loadingFinished());
	}

	ct::Projection CtVolume::getProjectionAt(size_t index) const {
		if (index < 0 || index >= _sinogram.size()) {
			throw std::out_of_range("Index out of bounds.");
		} else {
			ct::Projection projection = _sinogram[index].getPublicProjection();
			convertTo32bit(projection.image);
			projection.image = normalizeImage(projection.image, _minMaxValues.first, _minMaxValues.second);
			return projection;
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

	cv::Mat CtVolume::getVolumeCrossSection(size_t index) const {
		if (_volume.size() == 0) return cv::Mat();
		//copy to local variable because member variable might change during execution
		Axis axis = _crossSectionAxis;
		if (index >= 0 && (axis == Axis::X && index < _xMax) || (axis == Axis::Y && index < _yMax) || (axis == Axis::Z && index < _zMax)) {
			if (_volume.size() > 0 && _volume[0].size() > 0 && _volume[0][0].size() > 0) {
				size_t uSize;
				size_t vSize;
				if (axis == Axis::X) {
					uSize = _yMax;
					vSize = _zMax;
				} else if (axis == Axis::Y) {
					uSize = _xMax;
					vSize = _zMax;
				} else {
					uSize = _xMax;
					vSize = _yMax;
				}

				cv::Mat result(vSize, uSize, CV_32FC1);
				float* ptr;
				if (axis == Axis::X) {
#pragma omp parallel for private(ptr)
					for (int row = 0; row < result.rows; ++row) {
						ptr = result.ptr<float>(row);
						for (int column = 0; column < result.cols; ++column) {
							ptr[column] = _volume[index][column][result.rows - 1 - row];
						}
					}
				} else if (axis == Axis::Y) {
#pragma omp parallel for private(ptr)
					for (int row = 0; row < result.rows; ++row) {
						ptr = result.ptr<float>(row);
						for (int column = 0; column < result.cols; ++column) {
							ptr[column] = _volume[column][index][result.rows - 1 - row];
						}
					}
				} else {
#pragma omp parallel for private(ptr)
					for (int row = 0; row < result.rows; ++row) {
						ptr = result.ptr<float>(row);
						for (int column = 0; column < result.cols; ++column) {
							ptr[column] = _volume[column][row][index];
						}
					}
				}
				return result;
			}
			return cv::Mat();
		} else {
			throw std::out_of_range("Index out of bounds.");
		}
	}

	void CtVolume::setCrossSectionIndex(size_t index) {
		if (index >= 0 && (_crossSectionAxis == Axis::X && index < _xMax) || (_crossSectionAxis == Axis::Y && index < _yMax) || (_crossSectionAxis == Axis::Z && index < _zMax)) {
			_crossSectionIndex = index;
		}
	}

	void CtVolume::setCrossSectionAxis(Axis axis) {
		_crossSectionAxis = axis;
		if (_crossSectionAxis == Axis::X) {
			_crossSectionIndex = _xMax / 2;
		} else if (_crossSectionAxis == Axis::Y) {
			_crossSectionIndex = _yMax / 2;
		} else {
			_crossSectionIndex = _zMax / 2;
		}
	}

	size_t CtVolume::getCrossSectionIndex() const {
		return _crossSectionIndex;
	}

	size_t CtVolume::getCrossSectionSize() const {
		if (_crossSectionAxis == Axis::X) {
			return _xMax;
		} else if (_crossSectionAxis == Axis::Y) {
			return _yMax;
		} else {
			return _zMax;
		}
	}

	Axis CtVolume::getCrossSectionAxis() const {
		return _crossSectionAxis;
	}

	void CtVolume::setVolumeBounds(double xFrom, double xTo, double yFrom, double yTo, double zFrom, double zTo) {
		std::lock_guard<std::mutex> lock(_exclusiveFunctionsMutex);
		_xFrom_float = std::max(0.0, std::min(1.0, xFrom));
		_xTo_float = std::max(_xFrom_float, std::min(1.0, xTo));
		_yFrom_float = std::max(0.0, std::min(1.0, yFrom));
		_yTo_float = std::max(_xFrom_float, std::min(1.0, yTo));
		_zFrom_float = std::max(0.0, std::min(1.0, zFrom));
		_zTo_float = std::max(_xFrom_float, std::min(1.0, zTo));
		if (_sinogram.size() > 0) updateBoundaries();
	}

	void CtVolume::reconstructVolume(FilterType filterType) {
		std::lock_guard<std::mutex> lock(_exclusiveFunctionsMutex);
		_stop = false;
		if (_sinogram.size() > 0) {
			//resize the volume to the correct size
			_volume = std::vector<std::vector<std::vector<float>>>(_xMax, std::vector<std::vector<float>>(_yMax, std::vector<float>(_zMax, 0)));
			//mesure time
			clock_t start = clock();
			//fill the volume
			if (reconstructionCore(filterType)) {
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
				_volume.clear();
			}
		} else {
			std::cout << "Volume was not reconstructed, because the sinogram seems to be empty. Please load some images first." << std::endl;
			if (_emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::error("Volume was not reconstructed, because the sinogram seems to be empty. Please load some images first.")));
		}
	}

	void CtVolume::saveVolumeToBinaryFile(std::string filename) const {
		std::lock_guard<std::mutex> lock(_exclusiveFunctionsMutex);
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

	bool CtVolume::readImages(std::ifstream& csvStream, std::string path, int imgCnt) {
		//now load the actual image files and their parameters
		std::cout << "Parsing config file" << std::endl;
		std::stringstream lineStream;
		std::stringstream conversionStream;
		std::string line;
		std::string field;
		std::string file;
		double angle;
		double heightOffset;
		int cnt = 0;
		size_t rows;
		size_t cols;
		double min = std::numeric_limits<double>::quiet_NaN();
		double max = std::numeric_limits<double>::quiet_NaN();
		while (std::getline(csvStream, line) && !_stop) {
			lineStream.str(line);
			lineStream.clear();
			std::getline(lineStream, file, '\t');
			std::getline(lineStream, field, '\t');
			conversionStream.str(field);
			conversionStream.clear();
			conversionStream >> angle;
			std::getline(lineStream, field);
			conversionStream.str(field);
			conversionStream.clear();
			conversionStream >> heightOffset;
			//load the image
			_sinogram.push_back(ct::CtVolume::Projection(path + file, angle, heightOffset));
			cv::Mat image = _sinogram[cnt].getImage();
			//check if everything is ok
			if (!image.data) {
				//if there is no image data
				_sinogram.clear();
				std::string msg = "Error loading the image \"" + path + file + "\" (line " + std::to_string(cnt + 9) + "). Maybe it does not exist, permissions are missing or the format is not supported.";
				std::cout << msg << std::endl;
				if (_emitSignals) emit(loadingFinished(CompletionStatus::error(msg.c_str())));
				return false;
			} else if (image.depth() != CV_8U && image.depth() != CV_16U && image.depth() != CV_32F) {
				//wrong depth
				_sinogram.clear();
				std::string msg = "Error loading the image \"" + path + file + "\". The image depth must be either 8bit, 16bit or 32bit.";
				std::cout << msg << std::endl;
				if (_emitSignals) emit(loadingFinished(CompletionStatus::error(msg.c_str())));
				return false;
			} else {
				//make sure that all images have the same size
				if (cnt == 0) {
					rows = image.rows;
					cols = image.cols;
				} else {
					if (image.rows != rows || image.cols != cols) {
						//if the image has a different size than the images before stop and reverse
						_sinogram.clear();
						std::string msg = "Error loading the image \"" + file + "\", its dimensions differ from the images before.";
						std::cout << msg << std::endl;
						if (_emitSignals) emit(loadingFinished(CompletionStatus::error(msg.c_str())));
						return false;
					}
				}
				//compute the min max values
				double lMin, lMax;
				cv::minMaxLoc(image, &lMin, &lMax);
				if (image.type() == CV_8U) {
					lMin /= std::pow(2, 8);
					lMax /= std::pow(2, 8);
				} else {
					lMin /= std::pow(2, 16);
					lMax /= std::pow(2, 16);
				}
				if (std::isnan(min) || lMin < min) min = lMin;
				if (std::isnan(max) || lMax > max) max = lMax;
				//set image width and image height
				_imageWidth = cols;
				_imageHeight = rows;
				//output
				double percentage = floor(double(cnt) / double(imgCnt) * 100 + 0.5);
				std::cout << "\r" << "Analysing images: " << percentage << "%";
				if (_emitSignals) emit(loadingProgress(percentage));
			}
			++cnt;
		}
		_minMaxValues = std::make_pair(float(min), float(max));
		std::cout << std::endl;
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
			//clockwise rotation requires rotation in negative direction and ccw rotation requires positive direction
			//refers to the rotatin of the "camera"
			if ((rotationDirection == "cw" && diff > 0) || (rotationDirection == "ccw" && diff < 0)) {
				for (int i = 0; i < _sinogram.size(); ++i) {
					_sinogram[i].angle *= -1;
				}
			}
		}
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

	cv::Mat CtVolume::prepareProjection(size_t index, FilterType filterType) const {
		cv::Mat image = _sinogram[index].getImage();
		if (image.data) {
			convertTo32bit(image);
			preprocessImage(image, filterType);
		}
		return image;
	}

	void CtVolume::preprocessImage(cv::Mat& image, FilterType filterType) const {
		applyLogScaling(image);
		applyFourierFilter(image, filterType);
		applyFeldkampWeight(image);
	}

	void CtVolume::convertTo32bit(cv::Mat& img) {
		CV_Assert(img.depth() == CV_8U || img.depth() == CV_16U || img.depth() == CV_32F);
		if (img.depth() == CV_8U) {
			img.convertTo(img, CV_32F, 1.0 / (float)pow(2, 8));
		} else if (img.depth() == CV_16U) {
			img.convertTo(img, CV_32F, 1.0 / (float)pow(2, 16));
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
		// -ln(x)
		cv::log(image, image);
		image *= -1;
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

	bool CtVolume::reconstructionCore(FilterType filterType) {
		double imageLowerBoundU = matToImageU(0);
		//-0.1 is for absolute edge cases where the u-coordinate could be exactly the last pixel.
		//The bilinear interpolation would still try to access the next pixel, which then wouldn't exist.
		//The use of std::floor and std::ceil instead of simple integer rounding would prevent this problem, but also be slower.
		double imageUpperBoundU = matToImageU(_imageWidth - 1 - 0.1);
		//inversed because of inversed v axis in mat/image coordinate system
		double imageLowerBoundV = matToImageV(_imageHeight - 1 - 0.1);
		double imageUpperBoundV = matToImageV(0);

		double volumeLowerBoundY = volumeToWorldY(0);
		double volumeUpperBoundY = volumeToWorldY(_yMax);
		double volumeLowerBoundZ = volumeToWorldZ(0);
		double volumeUpperBoundZ = volumeToWorldZ(_zMax);

		//for the preloading of the next projection
		std::future<cv::Mat> future;

		for (int projection = 0; projection < _sinogram.size(); ++projection) {
			if (_stop) {
				std::cout << std::endl << "User interrupted. Stopping." << std::endl;
				if (_emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::interrupted()));
				return false;
			}
			//output percentage
			double percentage = floor((double)projection / (double)_sinogram.size() * 100 + 0.5);
			std::cout << "\r" << "Backprojecting: " << percentage << "%";
			if (_emitSignals) emit(reconstructionProgress(percentage, getVolumeCrossSection(_crossSectionIndex)));
			double beta_rad = (_sinogram[projection].angle / 180.0) * M_PI;
			double sine = sin(beta_rad);
			double cosine = cos(beta_rad);
			//load the projection, the projection for the next iteration is already prepared in a background thread
			cv::Mat image;
			if (projection == 0) {
				image = prepareProjection(projection, filterType);
			} else {
				image = future.get();
			}
			if (projection + 1 != _sinogram.size()) {
				future = std::async(std::launch::async, &CtVolume::prepareProjection, this, projection + 1, filterType);
			}
			//check if the image is good
			if (!image.data) {
				std::string msg = "The image " + _sinogram[projection].imagePath + " could not be accessed. Maybe it doesn't exist or has an unsupported format.";
				std::cout << std::endl << msg << std::endl;
				if (_emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::error(msg.c_str())));
				return false;
			}
			//copy some member variables to local variables, performance is better this way
			double heightOffset = _sinogram[projection].heightOffset;
			double uOffset = _uOffset;
			double SD = _SD;
			double radiusSquared = std::pow((_xSize / 2.0) - 3, 2);
#pragma omp parallel for schedule(dynamic)
			for (long xIndex = 0; xIndex < _xMax; ++xIndex) {
				double x = volumeToWorldX(xIndex);
				for (double y = volumeLowerBoundY; y < volumeUpperBoundY; ++y) {
					if ((x*x + y*y) < radiusSquared) {
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
								_volume[xIndex][worldToVolumeY(y)][worldToVolumeZ(z)] += bilinearInterpolation(u - double(u0), v - double(v0), u0v0, u1v0, u0v1, u1v1);
							}
						}
					}
				}
			}
		}
		std::cout << std::endl;
		return true;
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
		//make sure the cross section index is valid
		if (_crossSectionAxis == Axis::X && _crossSectionIndex >= _xMax) {
			_crossSectionIndex = _xMax / 2;
		} else if (_crossSectionAxis == Axis::Y && _crossSectionIndex >= _yMax) {
			_crossSectionIndex = _yMax / 2;
		} else if (_crossSectionAxis == Axis::Z && _crossSectionIndex >= _zMax) {
			_crossSectionIndex = _zMax / 2;
		}
		//precompute some values for faster processing
		_worldToVolumeXPrecomputed = (double(_xSize) / 2.0) - double(_xFrom);
		_worldToVolumeYPrecomputed = (double(_ySize) / 2.0) - double(_yFrom);
		_worldToVolumeZPrecomputed = (double(_zSize) / 2.0) - double(_zFrom);
		_volumeToWorldXPrecomputed = (double(_xSize) / 2.0) - double(_xFrom);
		_imageToMatUPrecomputed = double(_imageWidth) / 2.0;
		_imageToMatVPrecomputed = double(_imageHeight) / 2.0;
	}

	inline double CtVolume::worldToVolumeX(double xCoord) const {
		return xCoord + _worldToVolumeXPrecomputed;
	}

	inline double CtVolume::worldToVolumeY(double yCoord) const {
		return yCoord + _worldToVolumeYPrecomputed;
	}

	inline double CtVolume::worldToVolumeZ(double zCoord) const {
		return zCoord + _worldToVolumeZPrecomputed;
	}

	inline double CtVolume::volumeToWorldX(double xCoord) const {
		return xCoord - _volumeToWorldXPrecomputed;
	}

	inline double CtVolume::volumeToWorldY(double yCoord) const {
		return yCoord - (double(_ySize) / 2.0) + double(_yFrom);
	}

	inline double CtVolume::volumeToWorldZ(double zCoord) const {
		return zCoord - (double(_zSize) / 2.0) + double(_zFrom);
	}

	inline double CtVolume::imageToMatU(double uCoord)const {
		return uCoord + _imageToMatUPrecomputed;
	}

	inline double CtVolume::imageToMatV(double vCoord)const {
		//factor -1 because of different z-axis direction
		return (-1)*vCoord + _imageToMatVPrecomputed;
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