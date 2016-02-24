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
	CtVolume::CtVolume() : activeCudaDevices({ 0 }) {
		QObject::connect(this, SIGNAL(cudaThreadProgressUpdate(double, int, bool)), this, SLOT(emitGlobalCudaProgress(double, int, bool)));
	}

	CtVolume::CtVolume(std::string csvFile) : activeCudaDevices({ 0 }) {
		QObject::connect(this, SIGNAL(cudaThreadProgressUpdate(double, int, bool)), this, SLOT(emitGlobalCudaProgress(double, int, bool)));
		this->sinogramFromImages(csvFile);
	}

	bool CtVolume::cudaAvailable() {
		return (cv::cuda::getCudaEnabledDeviceCount() > 0);
	}

	void CtVolume::sinogramFromImages(std::string csvFile) {
		std::lock_guard<std::mutex> lock(this->exclusiveFunctionsMutex);
		this->stopActiveProcess = false;
		this->volume.clear();
		//delete the contents of the sinogram
		this->sinogram.clear();
		//open the csv file
		std::ifstream stream(csvFile.c_str(), std::ios::in);
		if (!stream.good()) {
			std::cerr << "Could not open CSV file - terminating" << std::endl;
			if (this->emitSignals) emit(loadingFinished(CompletionStatus::error("Could not open the config file.")));
			return;
		}
		//count the lines in the file
		int lineCnt = std::count(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>(), '\n') + 1;
		int imgCnt = lineCnt - 8;

		if (imgCnt <= 0) {
			std::cout << "CSV file does not contain any images." << std::endl;
			if (this->emitSignals) emit(loadingFinished(CompletionStatus::error("Apparently the config file does not contain any images.")));
			return;
		} else {
			//resize the sinogram to the correct size
			this->sinogram.reserve(imgCnt);

			//go back to the beginning of the file
			stream.seekg(std::ios::beg);

			//read the parameter section of the csv file
			std::string path;
			std::string rotationDirection;
			this->readParameters(stream, path, rotationDirection);

			//the image path might be relative
			path = this->glueRelativePath(csvFile, path);

			//read the images from the csv file
			if (!this->readImages(stream, path, imgCnt)) return;
			if (this->stopActiveProcess) {
				this->sinogram.clear();
				std::cout << "User interrupted. Stopping.";
				if (this->emitSignals) emit(loadingFinished(CompletionStatus::interrupted()));
				return;
			}

			if (this->sinogram.size() > 0) {
				//make the height offset values realtive
				this->makeHeightOffsetRelative();
				//make sure the rotation direction is correct
				this->correctAngleDirection(rotationDirection);
				//Axes: breadth = x, width = y, height = z
				this->xSize = this->imageWidth;
				this->ySize = this->imageWidth;
				this->zSize = this->imageHeight;
				this->updateBoundaries();
				switch (this->crossSectionAxis) {
					case Axis::X:
						this->crossSectionIndex = this->xMax / 2;
						break;
					case Axis::Y:
						this->crossSectionIndex = this->yMax / 2;
						break;
					case Axis::Z:
						this->crossSectionIndex = this->zMax / 2;
						break;
				}
				if (this->stopActiveProcess) {
					this->sinogram.clear();
					std::cout << "User interrupted. Stopping.";
					if (this->emitSignals) emit(loadingFinished(CompletionStatus::interrupted()));
					return;
				}
			}
		}
		if (this->emitSignals) emit(loadingFinished());
	}

	ct::Projection CtVolume::getProjectionAt(size_t index) const {
		if (index < 0 || index >= this->sinogram.size()) {
			throw std::out_of_range("Index out of bounds.");
		} else {
			ct::Projection projection = this->sinogram[index].getPublicProjection();
			convertTo32bit(projection.image);
			projection.image = this->normalizeImage(projection.image, this->minMaxValues.first, this->minMaxValues.second);
			return projection;
		}
	}

	size_t CtVolume::getSinogramSize() const {
		if (this->sinogram.size() > 0) {
			return this->sinogram.size();
		}
		return 0;
	}

	size_t CtVolume::getImageWidth() const {
		return this->imageWidth;
	}

	size_t CtVolume::getImageHeight() const {
		return this->imageHeight;
	}

	size_t CtVolume::getXSize() const {
		return this->xMax;
	}

	size_t CtVolume::getYSize() const {
		return this->yMax;
	}

	size_t CtVolume::getZSize() const {
		return this->zMax;
	}

	double CtVolume::getUOffset() const {
		return this->uOffset;
	}

	double CtVolume::getVOffset() const {
		return this->vOffset;
	}

	double CtVolume::getPixelSize() const {
		return this->pixelSize;
	}

	double CtVolume::getSO() const {
		return this->SO;
	}

	double CtVolume::getSD() const {
		return this->SD;
	}

	cv::Mat CtVolume::getVolumeCrossSection(size_t index) const {
		if (this->volume.size() == 0) return cv::Mat();
		//copy to local variable because member variable might change during execution
		Axis axis = this->crossSectionAxis;
		if (index >= 0 && (axis == Axis::X && index < this->xMax) || (axis == Axis::Y && index < this->yMax) || (axis == Axis::Z && index < this->zMax)) {
			if (this->volume.size() > 0 && this->volume[0].size() > 0 && this->volume[0][0].size() > 0) {
				size_t uSize;
				size_t vSize;
				if (axis == Axis::X) {
					uSize = this->yMax;
					vSize = this->zMax;
				} else if (axis == Axis::Y) {
					uSize = this->xMax;
					vSize = this->zMax;
				} else {
					uSize = this->xMax;
					vSize = this->yMax;
				}

				cv::Mat result(vSize, uSize, CV_32FC1);
				float* ptr;
				if (axis == Axis::X) {
#pragma omp parallel for private(ptr)
					for (int row = 0; row < result.rows; ++row) {
						ptr = result.ptr<float>(row);
						for (int column = 0; column < result.cols; ++column) {
							ptr[column] = this->volume[index][column][result.rows - 1 - row];
						}
					}
				} else if (axis == Axis::Y) {
#pragma omp parallel for private(ptr)
					for (int row = 0; row < result.rows; ++row) {
						ptr = result.ptr<float>(row);
						for (int column = 0; column < result.cols; ++column) {
							ptr[column] = this->volume[column][index][result.rows - 1 - row];
						}
					}
				} else {
#pragma omp parallel for private(ptr)
					for (int row = 0; row < result.rows; ++row) {
						ptr = result.ptr<float>(row);
						for (int column = 0; column < result.cols; ++column) {
							ptr[column] = this->volume[column][row][index];
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
		if (index >= 0 && (this->crossSectionAxis == Axis::X && index < this->xMax) || (this->crossSectionAxis == Axis::Y && index < this->yMax) || (this->crossSectionAxis == Axis::Z && index < this->zMax)) {
			this->crossSectionIndex = index;
		}
	}

	void CtVolume::setCrossSectionAxis(Axis axis) {
		this->crossSectionAxis = axis;
		if (this->crossSectionAxis == Axis::X) {
			this->crossSectionIndex = this->xMax / 2;
		} else if (this->crossSectionAxis == Axis::Y) {
			this->crossSectionIndex = this->yMax / 2;
		} else {
			this->crossSectionIndex = this->zMax / 2;
		}
	}

	size_t CtVolume::getCrossSectionIndex() const {
		return this->crossSectionIndex;
	}

	size_t CtVolume::getCrossSectionSize() const {
		if (this->crossSectionAxis == Axis::X) {
			return this->xMax;
		} else if (this->crossSectionAxis == Axis::Y) {
			return this->yMax;
		} else {
			return this->zMax;
		}
	}

	Axis CtVolume::getCrossSectionAxis() const {
		return this->crossSectionAxis;
	}

	bool CtVolume::getUseCuda() const {
		return this->useCuda;
	}

	std::vector<int> CtVolume::getActiveCudaDevices() const {
		return this->activeCudaDevices;
	}

	std::vector<std::string> CtVolume::getCudaDeviceList() const {
		int deviceCnt = cv::cuda::getCudaEnabledDeviceCount();
		//return an empty vector if there are no cuda devices
		if(deviceCnt < 1) return std::vector<std::string>();
		std::vector<std::string> result(deviceCnt);
		for (int i = 0; i < deviceCnt; ++i) {
			result[i] = ct::cuda::getDeviceName(i);
		}
		return result;
	}

	size_t CtVolume::getGpuSpareMemory() const {
		return this->gpuSpareMemory;
	}

	size_t CtVolume::getRequiredMemoryUpperBound() const {
		//size of volume
		size_t requiredMemory = this->xMax * this->yMax * this->zMax * sizeof(float);
		if (this->useCuda) {
			//images in RAM
			std::vector<int> devices = this->getActiveCudaDevices();
			requiredMemory += this->imageWidth * this->imageHeight * sizeof(float) * devices.size();
			for (int& device : devices) {
				cudaSetDevice(device);
				size_t maxSlices = this->getMaxChunkSize();
				//upper bounded by zMax
				//assume the volume is equally divided amonst gpus for simplicity
				maxSlices = std::min(maxSlices, size_t(std::ceil(double(this->zMax)/double(devices.size()))));
				//add extra memory required during download
				requiredMemory += this->xMax * this->yMax * maxSlices * sizeof(float);
			}
		} else {
			//images in RAM
			requiredMemory += this->imageWidth*this->imageHeight*sizeof(float) * 2;
		}
		return requiredMemory;
	}

	void CtVolume::setVolumeBounds(double xFrom, double xTo, double yFrom, double yTo, double zFrom, double zTo) {
		std::lock_guard<std::mutex> lock(this->exclusiveFunctionsMutex);
		this->xFrom_float = std::max(0.0, std::min(1.0, xFrom));
		this->xTo_float = std::max(this->xFrom_float, std::min(1.0, xTo));
		this->yFrom_float = std::max(0.0, std::min(1.0, yFrom));
		this->yTo_float = std::max(this->xFrom_float, std::min(1.0, yTo));
		this->zFrom_float = std::max(0.0, std::min(1.0, zFrom));
		zTo_float = std::max(this->xFrom_float, std::min(1.0, zTo));
		if (this->sinogram.size() > 0) this->updateBoundaries();
	}

	void CtVolume::setUseCuda(bool value) {
		this->useCuda = value;
	}

	void CtVolume::setActiveCudaDevices(std::vector<int> devices) {
		int deviceCnt = cv::cuda::getCudaEnabledDeviceCount();
		for (std::vector<int>::iterator i = devices.begin(); i != devices.end();) {
			if (*i >= deviceCnt) {
				i = devices.erase(i);
			} else {
				++i;
			}
		}
		if (devices.size() > 0) {
			this->activeCudaDevices = devices;
		} else {
			std::cout << "Active CUDA devices were not set because vector did not contain any valid device ID." << std::endl;
		}
	}

	void CtVolume::setGpuSpareMemory(size_t amount) {
		this->gpuSpareMemory = amount;
	}

	void CtVolume::setFrequencyFilterType(FilterType filterType) {
		this->filterType = filterType;
	}

	void CtVolume::reconstructVolume() {
		std::lock_guard<std::mutex> lock(this->exclusiveFunctionsMutex);

		this->stopActiveProcess = false;
		if (this->sinogram.size() > 0) {
			//clear potential old volume
			this->volume.clear();
			try {
			//resize the volume to the correct size
				this->volume = std::vector<std::vector<std::vector<float>>>(this->xMax, std::vector<std::vector<float>>(this->yMax, std::vector<float>(this->zMax, 0)));
			} catch (...) {
				std::cout << "The memory allocation for the volume failed. Maybe there is not enought free RAM." << std::endl;
				if (this->emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::error("The memory allocation for the volume failed. Maybe there is not enought free RAM.")));
				return;
			}
			//mesure time
			clock_t start = clock();
			//fill the volume
			bool result = false;
			if (this->useCuda && !this->cudaAvailable()) {
				std::cout << "CUDA processing was requested, but no capable CUDA device could be found. Falling back to CPU." << std::endl;
			}
			if (this->useCuda && this->cudaAvailable()) {
				result = this->launchCudaThreads();
			} else {
				result = this->reconstructionCore();
			}
			if (result) {
				//now fill the corners around the cylinder with the lowest density value
				double smallestValue = std::numeric_limits<double>::infinity();
				if (this->xMax > 0 && this->yMax > 0 && this->zMax > 0) {
#pragma omp parallel
				{
					double threadMin = std::numeric_limits<double>::infinity();
#pragma omp for schedule(dynamic)
					for (int x = 0; x < this->xMax; ++x) {
						for (int y = 0; y < this->yMax; ++y) {
							for (int z = 0; z < this->zMax; ++z) {
								if (this->volume[x][y][z] < threadMin) {
									threadMin = this->volume[x][y][z];
								}
							}
						}
					}
#pragma omp critical(compareLocalMinimums)
					{
						if (threadMin < smallestValue) smallestValue = threadMin;
					}
				}
#pragma omp parallel for schedule(dynamic)
					for (int x = 0; x < this->xMax; ++x) {
						for (int y = 0; y < this->yMax; ++y) {
							if (sqrt(this->volumeToWorldX(x)*this->volumeToWorldX(x) + this->volumeToWorldY(y)*this->volumeToWorldY(y)) >= ((double)this->xSize / 2) - 3) {
								for (int z = 0; z < this->zMax; ++z) {
									this->volume[x][y][z] = smallestValue;
								}
							}
						}
					}
				}

				//mesure time
				clock_t end = clock();
				std::cout << std::endl << "Volume successfully reconstructed (" << (double)(end - start) / CLOCKS_PER_SEC << "s)" << std::endl;
				if (this->emitSignals) emit(reconstructionFinished(this->getVolumeCrossSection(this->crossSectionIndex)));
			} else {
				this->volume.clear();

			}
		} else {
			std::cout << "Volume was not reconstructed, because the sinogram seems to be empty. Please load some images first." << std::endl;
			if (this->emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::error("Volume was not reconstructed, because the sinogram seems to be empty. Please load some images first.")));
		}
	}

	void CtVolume::saveVolumeToBinaryFile(std::string filename) const {
		std::lock_guard<std::mutex> lock(this->exclusiveFunctionsMutex);
		this->stopActiveProcess = false;
		if (this->volume.size() > 0 && this->volume[0].size() > 0 && this->volume[0][0].size() > 0) {
			{
				//write binary file
				QFile file(filename.c_str());
				if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
					std::cout << "Could not open the file. Maybe your path does not exist. No files were written." << std::endl;
					if (this->emitSignals) emit(savingFinished(CompletionStatus::error("Could not open the file. Maybe your path does not exist. No files were written.")));
					return;
				}
				QDataStream out(&file);
				out.setFloatingPointPrecision(QDataStream::SinglePrecision);
				out.setByteOrder(QDataStream::LittleEndian);
				//iterate through the volume
				for (int x = 0; x < this->xMax; ++x) {
					if (this->stopActiveProcess) break;
					if (this->emitSignals) {
						double percentage = floor(double(x) / double(this->volume.size()) * 100 + 0.5);
						emit(savingProgress(percentage));
					}
					for (int y = 0; y < this->yMax; ++y) {
						for (int z = 0; z < this->zMax; ++z) {
							//save one float of data
							out << this->volume[x][y][z];
						}
					}
				}
				file.close();
			}
			{
				//write information file
				QFileInfo fileInfo(filename.c_str());
				QString infoFileName = QDir(fileInfo.path()).absoluteFilePath(fileInfo.baseName().append(".txt"));
				//to circumvent naming conflicts with existing files
				if (QFileInfo(infoFileName).exists()) {
					unsigned int number = 1;
					do {
						infoFileName = QDir(fileInfo.path()).absoluteFilePath(fileInfo.baseName().append(QString::number(number)).append(".txt"));
						++number;
					} while (QFileInfo(infoFileName).exists());
				}
				QFile file(infoFileName);
				if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
					std::cout << "Could not write the info file." << std::endl;
					if (this->emitSignals) emit(savingFinished(CompletionStatus::error("Could not write the info file.")));
					return;
				}
				QTextStream out(&file);
				out << fileInfo.fileName() << endl << endl;
				out << "[Image dimensions]" << endl;
				out << "U resolution:\t" << this->imageWidth << endl;
				out << "V resolution:\t" << this->imageHeight << endl << endl;
				out << "[Reconstruction parameters]" << endl;
				out << "SD:\t\t\t\t" << this->SD << endl;
				out << "Pixel size:\t\t" << this->pixelSize << endl;
				out << "U offset:\t\t" << this->uOffset << endl;
				out << "X range:\t\t[" << this->xFrom << ".." << this->xTo << "]" << endl;
				out << "Y range:\t\t[" << this->yFrom << ".." << this->yTo << "]" << endl;
				out << "Z range:\t\t[" << this->zFrom << ".." << this->zTo << "]" << endl << endl;
				out << "[Volume dimensions]" << endl;
				out << "X size:\t\t\t" << this->xMax << endl;
				out << "Y size:\t\t\t" << this->yMax << endl;
				out << "Z size:\t\t\t" << this->zMax << endl << endl;
				out << "[Data format]" << endl;
				out << "Data type:\t\t32bit IEEE 754 float" << endl;
				out << "Endianness:\t\tLittle Endian" << endl;
				out << "Index order:\tZ fastest";
				file.close();
			}
			if (this->stopActiveProcess) {
				std::cout << "User interrupted. Stopping." << std::endl;
				if (this->emitSignals) emit(savingFinished(CompletionStatus::interrupted()));
				return;
			}
			std::cout << "Volume successfully saved." << std::endl;
			if (this->emitSignals) emit(savingFinished());
		} else {
			std::cout << "Did not save the volume, because it appears to be empty." << std::endl;
			if (this->emitSignals) emit(savingFinished(CompletionStatus::error("Did not save the volume, because it appears to be empty.")));
		}
	}

	void CtVolume::stop() {
		this->stopActiveProcess = true;
	}

	void CtVolume::setEmitSignals(bool value) {
		this->emitSignals = value;
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
		lineStream >> this->pixelSize;
		std::getline(stream, line);
		lineStream.str(line);
		lineStream.clear();
		std::getline(lineStream, rotationDirection, '\t');
		std::getline(stream, line);
		lineStream.str(line);
		lineStream.clear();
		lineStream >> this->uOffset;
		std::getline(stream, line);
		lineStream.str(line);
		lineStream.clear();
		lineStream >> this->vOffset;
		std::getline(stream, line);
		lineStream.str(line);
		lineStream.clear();
		lineStream >> this->SO;
		std::getline(stream, line);
		lineStream.str(line);
		lineStream.clear();
		lineStream >> this->SD;
		//leave out one line
		stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

		//convert the distance
		this->SD /= this->pixelSize;
		this->SO /= this->pixelSize;
		//convert uOffset and vOffset
		this->uOffset /= this->pixelSize;
		this->vOffset /= this->pixelSize;
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
		while (std::getline(csvStream, line) && !this->stopActiveProcess) {
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
			this->sinogram.push_back(ct::CtVolume::Projection(path + file, angle, heightOffset));
			cv::Mat image = this->sinogram[cnt].getImage();
			//check if everything is ok
			if (!image.data) {
				//if there is no image data
				this->sinogram.clear();
				std::string msg = "Error loading the image \"" + path + file + "\" (line " + std::to_string(cnt + 9) + "). Maybe it does not exist, permissions are missing or the format is not supported.";
				std::cout << msg << std::endl;
				if (this->emitSignals) emit(loadingFinished(CompletionStatus::error(msg.c_str())));
				return false;
			} else if (image.depth() != CV_8U && image.depth() != CV_16U && image.depth() != CV_32F) {
				//wrong depth
				this->sinogram.clear();
				std::string msg = "Error loading the image \"" + path + file + "\". The image depth must be either 8bit, 16bit or 32bit.";
				std::cout << msg << std::endl;
				if (this->emitSignals) emit(loadingFinished(CompletionStatus::error(msg.c_str())));
				return false;
			} else {
				//make sure that all images have the same size
				if (cnt == 0) {
					rows = image.rows;
					cols = image.cols;
				} else {
					if (image.rows != rows || image.cols != cols) {
						//if the image has a different size than the images before stop and reverse
						this->sinogram.clear();
						std::string msg = "Error loading the image \"" + file + "\", its dimensions differ from the images before.";
						std::cout << msg << std::endl;
						if (this->emitSignals) emit(loadingFinished(CompletionStatus::error(msg.c_str())));
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
				this->imageWidth = cols;
				this->imageHeight = rows;
				//output
				double percentage = floor(double(cnt) / double(imgCnt) * 100 + 0.5);
				std::cout << "\r" << "Analysing images: " << percentage << "%";
				if (this->emitSignals) emit(loadingProgress(percentage));
			}
			++cnt;
		}
		this->minMaxValues = std::make_pair(float(min), float(max));
		std::cout << std::endl;
		return true;
	}

	void CtVolume::makeHeightOffsetRelative() {
		//convert the heightOffset to a realtive value
		double sum = 0;
		for (int i = 0; i < this->sinogram.size(); ++i) {
			sum += this->sinogram[i].heightOffset;
		}
		sum /= (double)this->sinogram.size();
		for (int i = 0; i < this->sinogram.size(); ++i) {
			this->sinogram[i].heightOffset -= sum;			//substract average
			this->sinogram[i].heightOffset /= this->pixelSize;		//convert to pixels
		}
	}

	void CtVolume::correctAngleDirection(std::string rotationDirection) {
		//make sure the direction of the rotation is correct
		if (this->sinogram.size() > 1) {	//only possible if there are at least 2 images
			double diff = this->sinogram[1].angle - this->sinogram[0].angle;
			//clockwise rotation requires rotation in negative direction and ccw rotation requires positive direction
			//refers to the rotatin of the "camera"
			if ((rotationDirection == "cw" && diff > 0) || (rotationDirection == "ccw" && diff < 0)) {
				for (int i = 0; i < this->sinogram.size(); ++i) {
					this->sinogram[i].angle *= -1;
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

	cv::Mat CtVolume::prepareProjection(size_t index) const {
		cv::Mat image = this->sinogram[index].getImage();
		if (image.data) {
			convertTo32bit(image);
			this->preprocessImage(image);
		}
		return image;
	}

	void CtVolume::preprocessImage(cv::Mat& image) const {
		applyLogScaling(image);
		applyFourierFilter(image, this->filterType);
		this->applyFeldkampWeight(image);
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
#pragma omp parallel for private(ptr)
		for (int r = 0; r < image.rows; ++r) {
			ptr = image.ptr<float>(r);
			for (int c = 0; c < image.cols; ++c) {
				ptr[c] = ptr[c] * W(this->SD, this->matToImageU(c), this->matToImageV(r));
			}
		}
	}

	inline double CtVolume::W(double D, double u, double v) {
		return D / sqrt(D*D + u*u + v*v);
	}

	void CtVolume::applyFourierFilter(cv::Mat& image, FilterType filterType) {
		cv::Mat freq;
		cv::dft(image, freq, cv::DFT_COMPLEX_OUTPUT | cv::DFT_ROWS);
		unsigned int nyquist = (freq.cols / 2) + 1;
		cv::Vec2f* ptr;
#pragma omp parallel for private(ptr)
		for (int row = 0; row < freq.rows; ++row) {
			ptr = freq.ptr<cv::Vec2f>(row);
			for (int column = 0; column < nyquist; ++column) {
				switch (filterType) {
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

	cv::cuda::GpuMat CtVolume::cudaPreprocessImage(cv::cuda::GpuMat image, cv::cuda::Stream& stream, bool& success) const {
		success = true;
		cv::cuda::GpuMat tmp1;
		cv::cuda::GpuMat tmp2;
		image.convertTo(tmp1, CV_32FC1, stream);
		cv::cuda::log(tmp1, tmp1, stream);
		tmp1.convertTo(tmp1, tmp1.type(), -1, stream);
		cv::cuda::dft(tmp1, tmp2, image.size(), cv::DFT_ROWS, stream);
		bool successLocal;
		ct::cuda::applyFrequencyFiltering(tmp2, int(this->filterType), cv::cuda::StreamAccessor::getStream(stream), successLocal);
		success = success && successLocal;
		cv::cuda::dft(tmp2, image, image.size(), cv::DFT_ROWS | cv::DFT_REAL_OUTPUT, stream);
		ct::cuda::applyFeldkampWeightFiltering(image, this->SD, this->matToImageUPreprocessed, this->matToImageVPreprocessed, cv::cuda::StreamAccessor::getStream(stream), successLocal);
		success = success && successLocal;
		return image;
	}

	bool CtVolume::reconstructionCore() {
		double imageLowerBoundU = this->matToImageU(0);
		//-0.1 is for absolute edge cases where the u-coordinate could be exactly the last pixel.
		//The bilinear interpolation would still try to access the next pixel, which then wouldn't exist.
		//The use of std::floor and std::ceil instead of simple integer rounding would prevent this problem, but also be slower.
		double imageUpperBoundU = this->matToImageU(this->imageWidth - 1 - 0.1);
		//inversed because of inversed v axis in mat/image coordinate system
		double imageLowerBoundV = this->matToImageV(this->imageHeight - 1 - 0.1);
		double imageUpperBoundV = this->matToImageV(0);

		double volumeLowerBoundY = this->volumeToWorldY(0);
		double volumeUpperBoundY = this->volumeToWorldY(this->yMax);
		double volumeLowerBoundZ = this->volumeToWorldZ(0);
		double volumeUpperBoundZ = this->volumeToWorldZ(this->zMax);

		//for the preloading of the next projection
		std::future<cv::Mat> future;

		for (int projection = 0; projection < this->sinogram.size(); ++projection) {
			if (this->stopActiveProcess) {
				std::cout << std::endl << "User interrupted. Stopping." << std::endl;
				if (this->emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::interrupted()));
				return false;
			}
			//output percentage
			double percentage = std::round((double)projection / (double)this->sinogram.size() * 100);
			std::cout << "\r" << "Backprojecting: " << percentage << "%";
			if (this->emitSignals) emit(reconstructionProgress(percentage, this->getVolumeCrossSection(this->crossSectionIndex)));
			double beta_rad = (this->sinogram[projection].angle / 180.0) * M_PI;
			double sine = sin(beta_rad);
			double cosine = cos(beta_rad);
			//load the projection, the projection for the next iteration is already prepared in a background thread
			cv::Mat image;
			if (projection == 0) {
				image = this->prepareProjection(projection);
			} else {
				image = future.get();
			}
			if (projection + 1 != this->sinogram.size()) {
				future = std::async(std::launch::async, &CtVolume::prepareProjection, this, projection + 1);
			}
			//check if the image is good
			if (!image.data) {
				std::string msg = "The image " + this->sinogram[projection].imagePath + " could not be accessed. Maybe it doesn't exist or has an unsupported format.";
				std::cout << std::endl << msg << std::endl;
				if (this->emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::error(msg.c_str())));
				return false;
			}
			//copy some member variables to local variables, performance is better this way
			double heightOffset = this->sinogram[projection].heightOffset;
			double uOffset = this->uOffset;
			double SD = this->SD;
			double radiusSquared = std::pow((this->xSize / 2.0) - 3, 2);
#pragma omp parallel for schedule(dynamic)
			for (long xIndex = 0; xIndex < this->xMax; ++xIndex) {
				double x = this->volumeToWorldX(xIndex);
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

								u = this->imageToMatU(u);
								v = this->imageToMatV(v);

								//get the 4 surrounding pixels for the bilinear interpolation (note: u and v are always positive)
								int u0 = u;
								int u1 = u0 + 1;
								int v0 = v;
								int v1 = v0 + 1;

								//check if all the pixels are inside the image (after the coordinate transformation) (probably not necessary)
								//if (u0 < this->imageWidth && u0 >= 0 && u1 < this->imageWidth && u1 >= 0 && v0 < this->imageHeight && v0 >= 0 && v1 < this->imageHeight && v1 >= 0) {

								float* row = image.ptr<float>(v0);
								float u0v0 = row[u0];
								float u1v0 = row[u1];
								row = image.ptr<float>(v1);
								float u0v1 = row[u0];
								float u1v1 = row[u1];
								this->volume[xIndex][this->worldToVolumeY(y)][this->worldToVolumeZ(z)] += bilinearInterpolation(u - double(u0), v - double(v0), u0v0, u1v0, u0v1, u1v1);
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

	bool CtVolume::cudaReconstructionCore(size_t threadZMin, size_t threadZMax, int deviceId) {

		cudaSetDevice(deviceId);
		//for cuda error handling
		bool success;

		//precomputing some values
		double imageLowerBoundU = this->matToImageU(0);
		//-0.1 is for absolute edge cases where the u-coordinate could be exactly the last pixel.
		//The bilinear interpolation would still try to access the next pixel, which then wouldn't exist.
		//The use of std::floor and std::ceil instead of simple integer rounding would prevent this problem, but also be slower.
		double imageUpperBoundU = this->matToImageU(this->imageWidth - 1 - 0.1);
		//inversed because of inversed v axis in mat/image coordinate system
		double imageLowerBoundV = this->matToImageV(this->imageHeight - 1 - 0.1);
		double imageUpperBoundV = this->matToImageV(0);
		double radiusSquared = std::pow((this->xSize / 2.0) - 3, 2);

		const size_t progressUpdateRate = std::max(this->sinogram.size() / 102, static_cast<unsigned long long>(1));

		//first upload two images, so the memory used will be taken into consideration
		//prepare and upload image 1
		cv::Mat image;
		cv::cuda::GpuMat gpuPrefetchedImage;
		cv::cuda::GpuMat gpuCurrentImage;
		image = this->sinogram[0].getImage();
		cv::cuda::Stream gpuPreprocessingStream;
		try {
			gpuPrefetchedImage.upload(image, gpuPreprocessingStream);
			gpuPrefetchedImage = this->cudaPreprocessImage(gpuPrefetchedImage, gpuPreprocessingStream, success);
			gpuPreprocessingStream.waitForCompletion();
		} catch (...) {
			this->lastCudaErrorMessage = "An error occured during preprocessing of the image on the GPU. Maybe there was insufficient VRAM. You can try increasing the GPU spare memory setting.";
			std::cout << std::endl << this->lastCudaErrorMessage << std::endl;
			stopCudaThreads = true;
			return false;
		}
		if (!success) {
			this->lastCudaErrorMessage = "An error occured during preprocessing of the image on the GPU. Maybe there was insufficient VRAM. You can try increasing the GPU spare memory setting.";
			std::cout << std::endl << this->lastCudaErrorMessage << std::endl;
			stopCudaThreads = true;
			return false;
		}

		size_t sliceCnt = getMaxChunkSize();
		size_t currentSlice = threadZMin;

		if (sliceCnt < 1) {
			//too little memory
			this->lastCudaErrorMessage = "The VRAM of one of the used GPUs is insufficient to run a reconstruction.";
			std::cout << this->lastCudaErrorMessage << std::endl;
			//stop also the other cuda threads
			stopCudaThreads = true;
			return false;
		}

		while (currentSlice < threadZMax) {

			size_t const lastSlice = std::min(currentSlice + sliceCnt, threadZMax);
			size_t const xDimension = this->xMax;
			size_t const yDimension = this->yMax;
			size_t const zDimension = lastSlice - currentSlice;

			std::cout << std::endl << "GPU" << deviceId << " processing [" << currentSlice << ".." << lastSlice << ")" << std::endl;

			//allocate volume part memory on gpu
			cudaPitchedPtr gpuVolumePtr = ct::cuda::create3dVolumeOnGPU(xDimension, yDimension, zDimension, success);

			if (!success) {
				this->lastCudaErrorMessage = "An error occured during allocation of memory for the volume in the VRAM. Maybe the amount of free VRAM was insufficient. You can try changing the GPU spare memory setting.";
				std::cout << std::endl << this->lastCudaErrorMessage << std::endl;
				stopCudaThreads = true;
				return false;
			}

			for (int projection = 0; projection < this->sinogram.size(); ++projection) {

				//if user interrupts
				if (this->stopActiveProcess) {
					std::cout << std::endl << "User interrupted. Stopping." << std::endl;
					ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
					return false;
				}

				//if this thread shall be stopped (probably because an error in another thread occured)
				if (this->stopCudaThreads) {
					ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
					return false;
				}

				//emit progress update
				if (projection % progressUpdateRate == 0) {
					double chunkFinished = (currentSlice - threadZMin)*this->xMax*this->yMax;
					double currentChunk = (lastSlice - currentSlice)*this->xMax*this->yMax * (double(projection) / double(this->sinogram.size()));
					double percentage = (chunkFinished + currentChunk) / ((threadZMax - threadZMin)*this->xMax*this->yMax);
					emit(this->cudaThreadProgressUpdate(percentage, deviceId, (projection == 0)));
				}

				{
					cv::cuda::GpuMat tmp = gpuCurrentImage;
					gpuCurrentImage = gpuPrefetchedImage;
					gpuPrefetchedImage = tmp;
				}

				double beta_rad = (this->sinogram[projection].angle / 180.0) * M_PI;
				double sine = sin(beta_rad);
				double cosine = cos(beta_rad);

				//start reconstruction with current image
				ct::cuda::startReconstruction(gpuCurrentImage,
											  gpuVolumePtr,
											  xDimension,
											  yDimension,
											  zDimension,
											  currentSlice,
											  radiusSquared,
											  sine,
											  cosine,
											  this->sinogram[projection].heightOffset,
											  this->uOffset,
											  this->SD,
											  imageLowerBoundU,
											  imageUpperBoundU,
											  imageLowerBoundV,
											  imageUpperBoundV,
											  this->volumeToWorldXPrecomputed,
											  this->volumeToWorldYPrecomputed,
											  this->volumeToWorldZPrecomputed,
											  this->imageToMatUPrecomputed,
											  this->imageToMatVPrecomputed,
											  success);

				if (!success) {
					this->lastCudaErrorMessage = "An error occured during the launch of a reconstruction kernel on the GPU.";
					stopCudaThreads = true;
					ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
					if (!success) this->lastCudaErrorMessage += std::string(" Some memory allocated in the VRAM could not be freed.");
					std::cout << std::endl << this->lastCudaErrorMessage << std::endl;
					return false;
				}

				//prepare and upload next image
				if (projection < this->sinogram.size() - 1) {
					image = this->sinogram[projection + 1].getImage();
					if (!image.data) {
						this->lastCudaErrorMessage = "The image " + this->sinogram[projection + 1].imagePath + " could not be accessed. Maybe it doesn't exist or has an unsupported format.";
						stopCudaThreads = true;
						ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
						if (!success) this->lastCudaErrorMessage += std::string(" Some memory allocated in the VRAM could not be freed.");
						std::cout << std::endl << this->lastCudaErrorMessage << std::endl;
						return false;
					}
					try {
						gpuPrefetchedImage.upload(image, gpuPreprocessingStream);
						gpuPrefetchedImage = this->cudaPreprocessImage(gpuPrefetchedImage, gpuPreprocessingStream, success);
						gpuPreprocessingStream.waitForCompletion();
					} catch (...) {
						this->lastCudaErrorMessage = "An error occured during preprocessing of the image on the GPU. Maybe there was insufficient VRAM. You can try increasing the GPU spare memory value.";
						stopCudaThreads = true;
						ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
						if (!success) this->lastCudaErrorMessage += std::string(" Some memory allocated in the VRAM could not be freed.");
						std::cout << std::endl << this->lastCudaErrorMessage << std::endl;
						return false;
					}
					if (!success) {
						this->lastCudaErrorMessage = "An error occured during preprocessing of the image on the GPU. Maybe there was insufficient VRAM. You can try increasing the GPU spare memory setting.";
						stopCudaThreads = true;
						ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
						if (!success) this->lastCudaErrorMessage += std::string(" Some memory allocated in the VRAM could not be freed.");
						std::cout << std::endl << this->lastCudaErrorMessage << std::endl;
						return false;
					}
				}
			}

			//donload the reconstructed volume part
			std::shared_ptr<float> reconstructedVolumePart = ct::cuda::download3dVolume(gpuVolumePtr, xDimension, yDimension, zDimension, success);

			if (!success) {
				this->lastCudaErrorMessage = "An error occured during download of a reconstructed volume part from the VRAM.";
				stopCudaThreads = true;
				ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
				if (!success) this->lastCudaErrorMessage += std::string(" Some memory allocated in the VRAM could not be freed.");
				std::cout << std::endl << this->lastCudaErrorMessage << std::endl;
				return false;
			}

			//copy volume part to vector
			this->copyFromArrayToVolume(reconstructedVolumePart, zDimension, currentSlice);

			//free volume part memory on gpu
			ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);

			if (!success) {
				this->lastCudaErrorMessage = "Error: memory allocated in the VRAM could not be freed.";
				std::cout << std::endl << this->lastCudaErrorMessage << std::endl;
				stopCudaThreads = true;
				return false;
			}

			currentSlice += sliceCnt;
		}

		std::cout << std::endl;
		std::cout << "GPU" << deviceId << " finished." << std::endl;

		emit(this->cudaThreadProgressUpdate(1, deviceId, true));

		return true;
	}

	bool CtVolume::launchCudaThreads() {
		this->stopCudaThreads = false;
		//default error message
		this->lastCudaErrorMessage = "An error during the CUDA reconstruction occured.";

		std::map<int, double> scalingFactors = this->getGpuWeights(this->activeCudaDevices);

		//clear progress
		this->cudaThreadProgress.clear();
		for (int i = 0; i < this->activeCudaDevices.size(); ++i) {
			this->cudaThreadProgress.insert(std::make_pair(activeCudaDevices[i], 0));
		}

		//create vector to store threads
		std::vector<std::future<bool>> threads(this->activeCudaDevices.size());
		//launch one thread for each part of the volume (weighted by the amount of multiprocessors)
		size_t currentSlice = 0;
		for (int i = 0; i < this->activeCudaDevices.size(); ++i) {
			size_t sliceCnt = std::round(scalingFactors[this->activeCudaDevices[i]] * double(zMax));
			threads[i] = std::async(std::launch::async, &CtVolume::cudaReconstructionCore, this, currentSlice, currentSlice + sliceCnt, this->activeCudaDevices[i]);
			currentSlice += sliceCnt;
		}
		//wait for threads to finish
		bool result = true;
		for (int i = 0; i < this->activeCudaDevices.size(); ++i) {
			result = result && threads[i].get();
		}

		if (!result) {
			if (this->stopActiveProcess) {
				if (this->emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::interrupted()));
			} else {
				if (this->emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::error(this->lastCudaErrorMessage.c_str())));
			}
		}

		return result;
	}

	std::map<int, double> CtVolume::getGpuWeights(std::vector<int> const& devices) const {
		//get amount of multiprocessors per GPU
		std::map<int, int> multiprocessorCnt;
		size_t totalMultiprocessorsCnt = 0;
		std::map<int, int> busWidth;
		size_t totalBusWidth = 0;
		for (int i = 0; i < devices.size(); ++i) {
			multiprocessorCnt[devices[i]] = ct::cuda::getMultiprocessorCnt(devices[i]);
			totalMultiprocessorsCnt += multiprocessorCnt[devices[i]];
			busWidth[devices[i]] = ct::cuda::getMemoryBusWidth(devices[i]);
			totalBusWidth += busWidth[devices[i]];
			std::cout << "GPU" << devices[i] << std::endl;
			cudaSetDevice(devices[i]);
			std::cout << "\tFree memory: " << double(ct::cuda::getFreeMemory()) / 1024 / 1024 / 1025 << " Gb" << std::endl;
			std::cout << "\tMultiprocessor count: " << multiprocessorCnt[devices[i]] << std::endl;
		}
		std::map<int, double> scalingFactors;
		double scalingFactorSum = 0;
		for (int i = 0; i < devices.size(); ++i) {
			double multiprocessorScalingFactor = (double(multiprocessorCnt[devices[i]]) / double(totalMultiprocessorsCnt));
			double busWidthScalingFactor = (double(busWidth[devices[i]]) / double(totalBusWidth));
			scalingFactors[devices[i]] = multiprocessorScalingFactor*busWidthScalingFactor;
			scalingFactorSum += scalingFactors[devices[i]];
		}
		for (int i = 0; i < devices.size(); ++i) {
			scalingFactors[devices[i]] /= scalingFactorSum;
		}
		return scalingFactors;
	}

	size_t CtVolume::getMaxChunkSize() const {
		if (this->sinogram.size() < 1) return 0;
		long long freeMemory = ct::cuda::getFreeMemory();
		//spare some VRAM for other applications
		freeMemory -= this->gpuSpareMemory * 1024 * 1024;
		//spare memory for intermediate images and dft result
		freeMemory -= sizeof(float)*(this->imageWidth*this->imageHeight * 3 + (this->imageWidth / 2 - 1)*this->imageHeight * 2);

		if (freeMemory < 0) return 0;

		size_t sliceSize = this->xMax * this->yMax * sizeof(float);
		if (sliceSize == 0) return 0;
		size_t sliceCnt = freeMemory / sliceSize;

		return sliceCnt;
	}

	void CtVolume::copyFromArrayToVolume(std::shared_ptr<float> arrayPtr, size_t zSize, size_t zOffset) {
#pragma omp parallel for schedule(dynamic)
		for (int x = 0; x < this->xMax; ++x) {
			for (int y = 0; y < this->yMax; ++y) {
				for (int z = 0; z < zSize; ++z) {
					//xSize * ySize * zCoord + xSize * yChoord + xCoord
					volume[x][y][z + zOffset] = arrayPtr.get()[z * this->xMax * this->yMax + y * this->xMax + x];
				}
			}
		}
	}

	void CtVolume::updateBoundaries() {
		//calcualte bounds (+0.5 for correct rouding)
		this->xFrom = this->xFrom_float * this->xSize + 0.5;
		this->xTo = this->xTo_float * this->xSize + 0.5;
		this->yFrom = this->yFrom_float * this->ySize + 0.5;
		this->yTo = this->yTo_float * this->ySize + 0.5;
		this->zFrom = this->zFrom_float * this->zSize + 0.5;
		this->zTo = zTo_float * this->zSize + 0.5;
		this->xMax = this->xTo - this->xFrom;
		this->yMax = this->yTo - this->yFrom;
		this->zMax = this->zTo - this->zFrom;
		//make sure the cross section index is valid
		if (this->crossSectionAxis == Axis::X && this->crossSectionIndex >= this->xMax) {
			this->crossSectionIndex = this->xMax / 2;
		} else if (this->crossSectionAxis == Axis::Y && this->crossSectionIndex >= this->yMax) {
			this->crossSectionIndex = this->yMax / 2;
		} else if (this->crossSectionAxis == Axis::Z && this->crossSectionIndex >= this->zMax) {
			this->crossSectionIndex = this->zMax / 2;
		}
		//precompute some values for faster processing
		this->worldToVolumeXPrecomputed = (double(this->xSize) / 2.0) - double(this->xFrom);
		this->worldToVolumeYPrecomputed = (double(this->ySize) / 2.0) - double(this->yFrom);
		this->worldToVolumeZPrecomputed = (double(this->zSize) / 2.0) - double(this->zFrom);
		this->volumeToWorldXPrecomputed = (double(this->xSize) / 2.0) - double(this->xFrom);
		this->volumeToWorldYPrecomputed = (double(this->ySize) / 2.0) - double(this->yFrom);
		this->volumeToWorldZPrecomputed = (double(this->zSize) / 2.0) - double(this->zFrom);
		this->imageToMatUPrecomputed = double(this->imageWidth) / 2.0;
		this->imageToMatVPrecomputed = double(this->imageHeight) / 2.0;
		this->matToImageUPreprocessed = double(this->imageWidth) / 2.0;
		this->matToImageVPreprocessed = double(this->imageHeight / 2.0);
	}

	inline double CtVolume::worldToVolumeX(double xCoord) const {
		return xCoord + this->worldToVolumeXPrecomputed;
	}

	inline double CtVolume::worldToVolumeY(double yCoord) const {
		return yCoord + this->worldToVolumeYPrecomputed;
	}

	inline double CtVolume::worldToVolumeZ(double zCoord) const {
		return zCoord + this->worldToVolumeZPrecomputed;
	}

	inline double CtVolume::volumeToWorldX(double xCoord) const {
		return xCoord - this->volumeToWorldXPrecomputed;
	}

	inline double CtVolume::volumeToWorldY(double yCoord) const {
		return yCoord - this->volumeToWorldYPrecomputed;
	}

	inline double CtVolume::volumeToWorldZ(double zCoord) const {
		return zCoord - this->volumeToWorldZPrecomputed;
	}

	inline double CtVolume::imageToMatU(double uCoord)const {
		return uCoord + this->imageToMatUPrecomputed;
	}

	inline double CtVolume::imageToMatV(double vCoord)const {
		//factor -1 because of different z-axis direction
		return (-1)*vCoord + this->imageToMatVPrecomputed;
	}

	inline double CtVolume::matToImageU(double uCoord)const {
		return uCoord - this->matToImageUPreprocessed;
	}

	inline double CtVolume::matToImageV(double vCoord)const {
		//factor -1 because of different z-axis direction
		return (-1)*vCoord + this->matToImageVPreprocessed;
	}

	//============================================== PRIVATE SLOTS ==============================================\\

	void CtVolume::emitGlobalCudaProgress(double percentage, int deviceId, bool emitCrossSection) {
		this->cudaThreadProgress[deviceId] = percentage;
		double totalProgress = 0;
		for (int i = 0; i < this->activeCudaDevices.size(); ++i) {
			totalProgress += this->cudaThreadProgress[this->activeCudaDevices[i]];
		}
		totalProgress /= this->cudaThreadProgress.size();
		totalProgress *= 100;
		std::cout << "\r" << "Total completion: " << std::round(totalProgress) << "%";
		if (this->emitSignals) {
			if (emitCrossSection) {
				emit(reconstructionProgress(totalProgress, this->getVolumeCrossSection(this->crossSectionIndex)));
			} else {
				emit(reconstructionProgress(totalProgress, cv::Mat()));
			}
		}
	}

}