#include "CtVolume.h"


namespace ct {

	//data type projection visible to the outside

	Projection::Projection() { }

	Projection::Projection(cv::Mat image, double angle, double heightOffset) : image(image), angle(angle), heightOffset(heightOffset) { }
	
	//internal data type projection

	CtVolume::Projection::Projection() { }

	CtVolume::Projection::Projection(QString imagePath, double angle, double heightOffset) : imagePath(imagePath), angle(angle), heightOffset(heightOffset) {
		//empty
	}

	cv::Mat CtVolume::Projection::getImage() const {
		cv::Mat image;
		if (!utility::isCharCompatible(this->imagePath)) {
			std::shared_ptr<std::vector<char>> buffer = utility::readFileIntoBuffer(this->imagePath);
			if (buffer->empty()) {
				return cv::Mat();
			}
			image = cv::imdecode(*buffer, CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
		} else {
			image = cv::imread(this->imagePath.toStdString(), CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
		}

		return image;
	}

	ct::Projection CtVolume::Projection::getPublicProjection() const {
		return ct::Projection(this->getImage(), angle, heightOffset);
	}

	//internal class FftFilter

	CtVolume::FftFilter::FftFilter(FftFilter const& other) {
		this->width = other.width;
		this->height = other.height;
		this->isGood = this->isGood && (CUFFT_SUCCESS == cufftPlanMany(&this->forwardPlan, 1, &this->width, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, this->height));
		this->isGood = this->isGood && (CUFFT_SUCCESS == cufftPlanMany(&this->inversePlan, 1, &this->width, NULL, 0, 0, NULL, 0, 0, CUFFT_C2R, this->height));
	}

	CtVolume::FftFilter::FftFilter(int width, int height) : width(width), height(height) {
		this->isGood = this->isGood && (CUFFT_SUCCESS == cufftPlanMany(&this->forwardPlan, 1, &this->width, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, this->height));
		this->isGood = this->isGood && (CUFFT_SUCCESS == cufftPlanMany(&this->inversePlan, 1, &this->width, NULL, 0, 0, NULL, 0, 0, CUFFT_C2R, this->height));
	}

	CtVolume::FftFilter::~FftFilter() {
		cufftDestroy(this->forwardPlan);
		cufftDestroy(this->inversePlan);
	}

	bool CtVolume::FftFilter::good() {
		return this->isGood;
	}

	size_t CtVolume::FftFilter::getWorkSize() const {
		size_t forwardFftSize, inverseFftSize;
		cufftGetSize(this->forwardPlan, &forwardFftSize);
		cufftGetSize(this->inversePlan, &inverseFftSize);
		return forwardFftSize + inverseFftSize;
	}

	void CtVolume::FftFilter::setStream(cudaStream_t stream, bool& success) {
		success = true;
		success = success && (CUFFT_SUCCESS == cufftSetStream(this->forwardPlan, stream));
		success = success && (CUFFT_SUCCESS == cufftSetStream(this->inversePlan, stream));
	}

	void CtVolume::FftFilter::applyForward(cv::cuda::GpuMat& imageIn, cv::cuda::GpuMat& dftSpectrumOut, bool& success) const {
		success = true;		
		if (imageIn.cols != this->width || imageIn.rows != this->height || imageIn.type() != CV_32FC1 || !imageIn.isContinuous()) {
			std::cout << "Input image for FFT filter does not meet requirements (correct width, height, type CV_32FC1 and continuous in memory)." << std::endl;
			success = false;
		}
		if (dftSpectrumOut.cols != this->width/2 + 1 || dftSpectrumOut.rows != this->height || dftSpectrumOut.type() != CV_32FC2 || !dftSpectrumOut.isContinuous()) {
			std::cout << "Output image for FFT filter does not meet requirements (correct width (N/2 + 1), height, type CV_32FC2 and continuous in memory)." << std::endl;
			success = false;
		}
		success = success && (CUFFT_SUCCESS == cufftExecR2C(this->forwardPlan, imageIn.ptr<cufftReal>(), dftSpectrumOut.ptr<cufftComplex>()));
	}

	void CtVolume::FftFilter::applyInverse(cv::cuda::GpuMat& dftSpectrumIn, cv::cuda::GpuMat& imageOut, bool& success) const {
		success = true;
		if (dftSpectrumIn.cols != this->width / 2 + 1 || dftSpectrumIn.rows != this->height || dftSpectrumIn.type() != CV_32FC2 || !dftSpectrumIn.isContinuous()) {
			std::cout << "Output image for FFT filter does not meet requirements (correct width (N/2 + 1), height, type CV_32FC2 and continuous in memory)." << std::endl;
			success = false;
			return;
		}
		if (imageOut.cols != this->width || imageOut.rows != this->height || imageOut.type() != CV_32FC1 || !imageOut.isContinuous()) {
			std::cout << "Input image for FFT filter does not meet requirements (correct width, height, type CV_32FC1 and continuous in memory)." << std::endl;
			success = false;
			return;
		}
		success = success && (CUFFT_SUCCESS == cufftExecC2R(this->inversePlan, dftSpectrumIn.ptr<cufftComplex>(), imageOut.ptr<cufftReal>()));
	}

	//============================================== PUBLIC ==============================================\\

	//constructor
	CtVolume::CtVolume() : activeCudaDevices({ 0 }) {
		this->initialise();
	}

	CtVolume::CtVolume(QString csvFile) : activeCudaDevices({ 0 }) {
		this->initialise();
		this->sinogramFromImages(csvFile);
	}

	bool CtVolume::cudaAvailable(bool verbose) {
		return (CtVolume::getCudaDeviceCount(verbose) > 0);
	}

	int CtVolume::getCudaDeviceCount(bool verbose) {
		int count;
		cudaError_t error = cudaGetDeviceCount(&count);

		if (error == cudaSuccess) {
			return count;
		} else if (error == cudaErrorInsufficientDriver && verbose) {
			std::cout << "No sufficient version of the Nvidia driver could be found. CUDA will not be avialable." << std::endl;
			return -1;
		} else if (error == cudaErrorNoDevice && verbose) {
			std::cout << "No CUDA devices could be found. CUDA will not be available." << std::endl;
		} else if (verbose) {
			std::cout << "An error occured while trying to retrieve CUDA devices. CUDA will not be available. Error: " << cudaGetErrorString(error) << std::endl;
		}
		return 0;
	}

	bool CtVolume::sinogramFromImages(QString csvFile) {
		std::lock_guard<std::mutex> lock(this->exclusiveFunctionsMutex);
		this->stopActiveProcess = false;
		this->volume.clear();
		//delete the contents of the sinogram
		this->sinogram.clear();
		//open the csv file
		QFile file(csvFile);
		if (!file.open(QIODevice::ReadOnly)) {
			std::cerr << "Could not open CSV file - terminating" << std::endl;
			if (this->emitSignals) emit(loadingFinished(CompletionStatus::error("Could not open the config file.")));
			return false;		
		}
		QTextStream in(&file);
		QDir imageDir;
		std::string rotationDirection;

		//read parameters
		{
			bool success;
			bool totalSuccess = true;
			imageDir = QDir(in.readLine().section('\t', 0, 0));
			this->pixelSize = in.readLine().section('\t', 0, 0).toDouble(&success);
			totalSuccess = totalSuccess && success;
			rotationDirection = in.readLine().section('\t', 0, 0).toStdString();
			this->uOffset = in.readLine().section('\t', 0, 0).toDouble(&success);
			totalSuccess = totalSuccess && success;
			this->vOffset = in.readLine().section('\t', 0, 0).toDouble(&success);
			totalSuccess = totalSuccess && success;
			this->SO = in.readLine().section('\t', 0, 0).toDouble(&success);
			totalSuccess = totalSuccess && success;
			this->SD = in.readLine().section('\t', 0, 0).toDouble(&success);
			totalSuccess = totalSuccess && success;
			//leave out one line
			in.readLine();
			//convert the distance
			this->SD /= this->pixelSize;
			this->SO /= this->pixelSize;
			//convert uOffset and vOffset
			this->uOffset /= this->pixelSize;
			this->vOffset /= this->pixelSize;
			if (!totalSuccess) {
				std::cout << "Could not read the parameters from the CSV file successfully." << std::endl;
				if (this->emitSignals) emit(loadingFinished(CompletionStatus::error("Could not read the parameters from the CSV file successfully.")));
				return false;
			}
		}

		//create image path
		if (imageDir.isRelative()) {
			imageDir = QDir(QDir::cleanPath(QFileInfo(csvFile).absoluteDir().absoluteFilePath(imageDir.path())));
		}

		//read images
		{
			bool success;
			bool totalSuccess = true;
			QString line;
			QString imageFilename;
			double angle;
			double heightOffset;
			while (!in.atEnd()) {
				line = in.readLine();
				QStringList fields = line.split('\t');
				if (fields.size() >= 2) {
					imageFilename = fields[0];
					angle = fields[1].toDouble(&success);
					totalSuccess = totalSuccess && success;
				}
				if (fields.size() >= 3) {
					heightOffset = fields[2].toDouble(&success);
					totalSuccess = totalSuccess && success;
				} else {
					heightOffset = 0;
				}
				//add the image
				this->sinogram.push_back(Projection(imageDir.absoluteFilePath(imageFilename), angle, heightOffset));
			}
			if (!totalSuccess) {
				std::cout << "Could not read the image parameters from the CSV file successfully." << std::endl;
				if (this->emitSignals) emit(loadingFinished(CompletionStatus::error("Could not read the image parameters from the CSV file successfully.")));
				return false;
			}
			if (this->sinogram.size() == 0) {
				std::cout << "Apparently the config file does not contain any images." << std::endl;
				if (this->emitSignals) emit(loadingFinished(CompletionStatus::error("Apparently the config file does not contain any images.")));
				return false;
			}
		}

		//load and check images
		{
			size_t cnt = 0;
			size_t rows, cols;
			double min = std::numeric_limits<double>::quiet_NaN();
			double max = std::numeric_limits<double>::quiet_NaN();
			for (Projection const& projection : this->sinogram) {
				if (this->stopActiveProcess) {
					this->sinogram.clear();
					std::cout << "User interrupted. Stopping.";
					if (this->emitSignals) emit(loadingFinished(CompletionStatus::interrupted()));
					return false;
				}
				cv::Mat image = projection.getImage();
				if (!image.data) {
					//if there is no image data
					this->sinogram.clear();
					QString msg = QString("Error loading the image \"%1\". Maybe it does not exist, permissions are missing or the format is not supported.").arg(projection.imagePath);
					std::cout << msg.toStdString() << std::endl;
					if (this->emitSignals) emit(loadingFinished(CompletionStatus::error(msg)));
					return false;
				} else if (image.depth() != CV_8U && image.depth() != CV_16U && image.depth() != CV_32F) {
					//wrong depth
					this->sinogram.clear();
					QString msg = QString("Error loading the image \"%1\". The image depth must be either 8bit, 16bit or 32bit.").arg(projection.imagePath);
					std::cout << msg.toStdString() << std::endl;
					if (this->emitSignals) emit(loadingFinished(CompletionStatus::error(msg)));
					return false;
				} else {
					//make sure that all images have the same size
					if (cnt == 0) {
						rows = image.rows;
						cols = image.cols;
						this->imageType = image.type();
					} else {
						if (image.rows != rows || image.cols != cols) {
							//if the image has a different size than the images before stop and reverse
							this->sinogram.clear();
							QString msg = QString("Error loading the image \"%1\", its dimensions differ from the images before.").arg(projection.imagePath);
							std::cout << msg.toStdString() << std::endl;
							if (this->emitSignals) emit(loadingFinished(CompletionStatus::error(msg)));
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
					double percentage = std::round(double(cnt) / double(this->sinogram.size()) * 100);
					std::cout << "\r" << "Analysing images: " << percentage << "%";
					if (this->emitSignals) emit(loadingProgress(percentage));
				}
				++cnt;
			}
			this->minMaxValues = std::make_pair(float(min), float(max));
			std::cout << std::endl;
		}

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
		if (this->emitSignals) emit(loadingFinished());
		return true;
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

	cv::Mat CtVolume::getVolumeCrossSection(Axis axis, size_t index) const {
		return this->volume.getVolumeCrossSection(axis, index, CoordinateSystemOrientation::LEFT_HANDED);
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

	bool CtVolume::getEmitSignals() const {
		return this->emitSignals;
	}

	bool CtVolume::getUseCuda() const {
		return this->useCuda;
	}

	std::vector<int> CtVolume::getActiveCudaDevices() const {
		return this->activeCudaDevices;
	}

	std::vector<std::string> CtVolume::getCudaDeviceList() const {
		int deviceCnt = CtVolume::getCudaDeviceCount();
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
			std::vector<int> devices = this->getActiveCudaDevices();
			//image in RAM + 2 * page locked memory
			requiredMemory += this->imageWidth * this->imageHeight * sizeof(float) * devices.size() * 3;
		} else {
			//images in RAM
			requiredMemory += this->imageWidth * this->imageHeight * sizeof(float) * 2;
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
		int deviceCnt = CtVolume::getCudaDeviceCount();
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

		//default error message
		this->lastErrorMessage = "An error during the reconstruction occured.";

		this->stopActiveProcess = false;
		if (this->sinogram.size() > 0) {
			//clear potential old volume
			this->volume.clear();
			try {
			//resize the volume to the correct size
				this->volume.reinitialise(this->xMax, this->yMax, this->zMax, 0);
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
				this->volume.setMemoryLayout(IndexOrder::X_FASTEST);
				result = this->launchCudaThreads();
			} else {
				this->volume.setMemoryLayout(IndexOrder::Z_FASTEST);
				result = this->reconstructionCore();
			}
			if (result) {
				//now fill the corners around the cylinder with the lowest density value
//				double smallestValue = std::numeric_limits<double>::infinity();
//				if (this->xMax > 0 && this->yMax > 0 && this->zMax > 0) {
//					double radiusSquared = std::pow(((double)this->xSize / 2) - 3, 2);
//#pragma omp parallel
//				{
//					double threadMin = std::numeric_limits<double>::infinity();
//#pragma omp for schedule(dynamic)
//					for (int x = 0; x < this->xMax; ++x) {
//						for (int y = 0; y < this->yMax; ++y) {
//							if (this->volumeToWorldX(x)*this->volumeToWorldX(x) + this->volumeToWorldY(y)*this->volumeToWorldY(y) <= radiusSquared) {
//								for (int z = 0; z < this->zMax; ++z) {
//									if (this->volume.at(x, y, z) < threadMin) {
//										threadMin = this->volume.at(x, y, z);
//									}
//								}
//							}
//						}
//					}
//#pragma omp critical(compareLocalMinimums)
//					{
//						if (threadMin < smallestValue) smallestValue = threadMin;
//					}
//				}
//#pragma omp parallel for schedule(dynamic)
//					for (int x = 0; x < this->xMax; ++x) {
//						for (int y = 0; y < this->yMax; ++y) {
//							if (this->volumeToWorldX(x)*this->volumeToWorldX(x) + this->volumeToWorldY(y)*this->volumeToWorldY(y) >= radiusSquared) {
//								for (int z = 0; z < this->zMax; ++z) {
//									this->volume.at(x, y, z) = smallestValue;
//								}
//							}
//						}
//					}
//				}

				//mesure time
				clock_t end = clock();
				std::cout << std::endl << "Volume successfully reconstructed (" << (double)(end - start) / CLOCKS_PER_SEC << "s)" << std::endl;
				if (this->emitSignals) emit(reconstructionFinished(this->getVolumeCrossSection(this->crossSectionAxis, this->crossSectionIndex)));
			} else {
				this->volume.clear();
				if (this->stopActiveProcess) {
					//user interrupted
					std::cout << std::endl << "User interrupted. Stopping." << std::endl;
					if (this->emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::interrupted()));
				} else {
					//there was an error
					std::cout << std::endl << this->lastErrorMessage << std::endl;
					if (this->emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::error(this->lastErrorMessage.c_str())));
				}
			}
		} else {
			std::cout << "Volume was not reconstructed, because the sinogram seems to be empty. Please load some images first." << std::endl;
			if (this->emitSignals) emit(reconstructionFinished(cv::Mat(), CompletionStatus::error("Volume was not reconstructed, because the sinogram seems to be empty. Please load some images first.")));
		}
	}

	void CtVolume::saveVolumeToBinaryFile(QString filename, IndexOrder indexOrder, QDataStream::ByteOrder byteOrder) const {
		std::lock_guard<std::mutex> lock(this->exclusiveFunctionsMutex);
		this->stopActiveProcess = false;

		//write information file
		QFileInfo fileInfo(filename);
		QString infoFileName = QDir(fileInfo.path()).absoluteFilePath(fileInfo.baseName().append(".txt"));
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
		if (byteOrder == QDataStream::LittleEndian) {
			out << "Byte order:\t\tLittle endian" << endl;
		} else {
			out << "Byte order:\t\tBig endian" << endl;
		}
		if (indexOrder == IndexOrder::Z_FASTEST) {
			out << "Index order:\tZ fastest";
		} else {
			out << "Index order:\tX fastest";
		}
		file.close();

		//write binary file
		this->volume.saveToBinaryFile(filename, indexOrder, QDataStream::SinglePrecision, byteOrder);

	}

	void CtVolume::stop() {
		this->stopActiveProcess = true;
		this->volume.stop();
	}

	void CtVolume::setEmitSignals(bool value) {
		this->emitSignals = value;
		this->volume.setEmitSignals(value);
	}

	//============================================== PRIVATE ==============================================\\

	void CtVolume::initialise() {
		qRegisterMetaType<CompletionStatus>("CompletionStatus");
		QObject::connect(this, SIGNAL(cudaThreadProgressUpdate(double, int, bool)), this, SLOT(emitGlobalCudaProgress(double, int, bool)));
		QObject::connect(&this->volume, SIGNAL(savingProgress(double)), this, SIGNAL(savingProgress(double)));
		QObject::connect(&this->volume, SIGNAL(savingFinished(CompletionStatus)), this, SIGNAL(savingFinished(CompletionStatus)));
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
		//images must be scaled in case different depths are mixed (-> equal value range)
		if (img.depth() == CV_8U) {
			img.convertTo(img, CV_32F, 1.0 / 255.0);
		} else if (img.depth() == CV_16U) {
			img.convertTo(img, CV_32F, 1.0 / 65535.0);
		}
	}

	void CtVolume::applyFeldkampWeight(cv::Mat& image) const {
		CV_Assert(image.channels() == 1);
		CV_Assert(image.depth() == CV_32F);

		float* ptr;
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
		for (int row = 0; row < freq.rows; ++row) {
			ptr = freq.ptr<cv::Vec2f>(row);
			for (unsigned int column = 0; column < nyquist; ++column) {
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

	void CtVolume::cudaPreprocessImage(cv::cuda::GpuMat& imageIn, cv::cuda::GpuMat& imageOut, cv::cuda::GpuMat& dftTmp, FftFilter& fftFilter, bool& success, cv::cuda::Stream& stream) const {
		success = true;
		bool successLocal;
		cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(stream);
		//images must be scaled in case different depths are mixed (-> equal value range)
		double scalingFactor = 1.0;
		if (imageIn.depth() == CV_8U) {
			scalingFactor = 255.0;
		} else if (imageIn.depth() == CV_16U) {
			scalingFactor = 65535.0;
		}
		//convert to 32bit
		imageIn.convertTo(imageOut, CV_32FC1, 1.0/scalingFactor, stream);
		//logarithmic scale
		cv::cuda::log(imageOut, imageOut, stream);
		//multiply by -1
		imageOut.convertTo(imageOut, imageOut.type(), -1, stream);
		//transform to frequency domain
		//cv::cuda::dft(imageOut, dftTmp, image.size(), cv::DFT_ROWS, stream);
		fftFilter.applyForward(imageOut, dftTmp, successLocal);
		success = success && successLocal;
		//apply frequency filter
		ct::cuda::applyFrequencyFiltering(dftTmp, int(this->filterType), cudaStream, successLocal);
		success = success && successLocal;
		//transform back to spatial domain
		//cv::cuda::dft(dftTmp, imageOut, image.size(), cv::DFT_ROWS | cv::DFT_REAL_OUTPUT, stream);
		fftFilter.applyInverse(dftTmp, imageOut, successLocal);
		success = success && successLocal;
		//apply the feldkamp weights
		ct::cuda::applyFeldkampWeightFiltering(imageOut, this->SD, this->matToImageUPreprocessed, this->matToImageVPreprocessed, cudaStream, successLocal);
		success = success && successLocal;
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
				return false;
			}
			//output percentage
			double percentage = std::round((double)projection / (double)this->sinogram.size() * 100);
			std::cout << "\r" << "Backprojecting: " << percentage << "%";
			if (this->emitSignals) emit(reconstructionProgress(percentage, this->getVolumeCrossSection(this->crossSectionAxis, this->crossSectionIndex)));
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
				this->lastErrorMessage = "The image " + this->sinogram[projection].imagePath.toStdString() + " could not be accessed. Maybe it doesn't exist or has an unsupported format.";
				return false;
			}
			//copy some member variables to local variables, performance is better this way
			double heightOffset = this->sinogram[projection].heightOffset;
			double uOffset = this->uOffset;
			double SD = this->SD;
			double radiusSquared = std::pow((this->xSize / 2.0) - 3, 2);
			float* volumePtr;
#pragma omp parallel for private(volumePtr) schedule(dynamic)
			for (long xIndex = 0; xIndex < this->xMax; ++xIndex) {
				double x = this->volumeToWorldX(xIndex);
				volumePtr = this->volume.slicePtr(xIndex);
				for (double y = volumeLowerBoundY; y < volumeUpperBoundY; ++y) {
					if ((x*x + y*y) >= radiusSquared) {
						volumePtr += this->zMax;
						continue;
					}
					//if the voxel is inside the reconstructable cylinder
					for (double z = volumeLowerBoundZ; z < volumeUpperBoundZ; ++z, ++volumePtr) {

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
							//this->volume.at(xIndex, this->worldToVolumeY(y), this->worldToVolumeZ(z)) += bilinearInterpolation(u - double(u0), v - double(v0), u0v0, u1v0, u0v1, u1v1);
							//size_t index = this->worldToVolumeY(y)*this->zMax + this->worldToVolumeZ(z);
							(*volumePtr) += bilinearInterpolation(u - double(u0), v - double(v0), u0v0, u1v0, u0v1, u1v1);
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

		try {

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

			const size_t progressUpdateRate = std::max(this->sinogram.size() / 102 / this->getActiveCudaDevices().size(), static_cast<size_t>(1));

			//image in RAM
			cv::Mat image;
			//page-locked RAM memeory for async upload
			std::vector<cv::cuda::HostMem> memory(2, cv::cuda::HostMem(this->imageHeight, this->imageWidth, this->imageType, cv::cuda::HostMem::PAGE_LOCKED));
			//image on gpu
			std::vector<cv::cuda::GpuMat> gpuImage(2);
			gpuImage[0] = cv::cuda::createContinuous(this->imageHeight, this->imageWidth, this->imageType);
			gpuImage[1] = cv::cuda::createContinuous(this->imageHeight, this->imageWidth, this->imageType);
			//streams for alternation
			std::vector<cv::cuda::Stream> stream(2);
			//temporary gpu mats for preprocessing
			std::vector<cv::cuda::GpuMat> preprocessedGpuImage(2);
			preprocessedGpuImage[0] = cv::cuda::createContinuous(this->imageHeight, this->imageWidth, CV_32FC1);
			preprocessedGpuImage[1] = cv::cuda::createContinuous(this->imageHeight, this->imageWidth, CV_32FC1);
			std::vector<cv::cuda::GpuMat> fftTmp(2);
			fftTmp[0] = cv::cuda::createContinuous(this->imageHeight, this->imageWidth / 2 + 1, CV_32FC2);
			fftTmp[1] = cv::cuda::createContinuous(this->imageHeight, this->imageWidth / 2 + 1, CV_32FC2);
			std::vector<FftFilter> fftFilter(2, FftFilter(this->imageWidth, this->imageHeight));
			if(!fftFilter[0].good() || !fftFilter[1].good()){
				this->lastErrorMessage = "An error occured during creation of the FFT filter. Maybe the amount of free VRAM was insufficient. You can try changing the GPU spare memory setting.";
				stopCudaThreads = true;
				return false;
			}
			{
				bool successLocal = true;
				fftFilter[0].setStream(cv::cuda::StreamAccessor::getStream(stream[0]), success);
				successLocal = successLocal && success;
				fftFilter[1].setStream(cv::cuda::StreamAccessor::getStream(stream[1]), success);
				successLocal = successLocal && success;
				if (!successLocal) {
					this->lastErrorMessage = "An error occured while assigning the streams for the FFT filters.";
					stopCudaThreads = true;
					return false;
				}
			}

			size_t sliceCnt = getMaxChunkSize(fftFilter[0].getWorkSize());
			size_t currentSlice = threadZMin;

			if (sliceCnt < 1) {
				//too little memory
				this->lastErrorMessage = "The VRAM of one of the used GPUs is insufficient to run a reconstruction.";
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
					this->lastErrorMessage = "An error occured during allocation of memory for the volume in the VRAM. Maybe the amount of free VRAM was insufficient. You can try changing the GPU spare memory setting.";
					stopCudaThreads = true;
					return false;
				}

				for (int projection = 0; projection < this->sinogram.size(); ++projection) {

					//if user interrupts
					if (this->stopActiveProcess) {
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
						double currentChunk = zDimension*this->xMax*this->yMax * (double(projection) / double(this->sinogram.size()));
						double percentage = (chunkFinished + currentChunk) / ((threadZMax - threadZMin)*this->xMax*this->yMax);
						emit(this->cudaThreadProgressUpdate(percentage, deviceId, (projection == 0)));
					}

					int current = projection % 2;

					double beta_rad = (this->sinogram[projection].angle / 180.0) * M_PI;
					double sine = sin(beta_rad);
					double cosine = cos(beta_rad);

					//prepare and upload next image
					image = this->sinogram[projection].getImage();
					if (!image.data) {
						this->lastErrorMessage = "The image " + this->sinogram[projection].imagePath.toStdString() + " could not be accessed. Maybe it doesn't exist or has an unsupported format.";
						stopCudaThreads = true;
						ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
						if (!success) this->lastErrorMessage += std::string(" Some memory allocated in the VRAM could not be freed.");
						return false;
					}
					try {
						stream[current].waitForCompletion();
						image.copyTo(memory[current]);
						gpuImage[current].upload(memory[current], stream[current]);
						this->cudaPreprocessImage(gpuImage[current], preprocessedGpuImage[current], fftTmp[current], fftFilter[current], success, stream[current]);
					} catch (...) {
						this->lastErrorMessage = "An error occured during preprocessing of the image on the GPU. Maybe there was insufficient VRAM. You can try increasing the GPU spare memory value.";
						stopCudaThreads = true;
						ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
						if (!success) this->lastErrorMessage += std::string(" Some memory allocated in the VRAM could not be freed.");
						return false;
					}
					if (!success) {
						this->lastErrorMessage = "An error occured during preprocessing of the image on the GPU. Maybe there was insufficient VRAM. You can try increasing the GPU spare memory setting.";
						stopCudaThreads = true;
						ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
						if (!success) this->lastErrorMessage += std::string(" Some memory allocated in the VRAM could not be freed.");
						return false;
					}

					//start reconstruction with current image
					ct::cuda::startReconstruction(preprocessedGpuImage[current],
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
												  cv::cuda::StreamAccessor::getStream(stream[current]),
												  success);

					if (!success) {
						this->lastErrorMessage = "An error occured during the launch of a reconstruction kernel on the GPU.";
						stopCudaThreads = true;
						ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
						if (!success) this->lastErrorMessage += std::string(" Some memory allocated in the VRAM could not be freed.");
						return false;
					}

				}

				//donload the reconstructed volume part
				ct::cuda::download3dVolume(gpuVolumePtr, this->volume.slicePtr(currentSlice), xDimension, yDimension, zDimension, success);

				if (!success) {
					this->lastErrorMessage = "An error occured during download of a reconstructed volume part from the VRAM.";
					stopCudaThreads = true;
					ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
					if (!success) this->lastErrorMessage += std::string(" Some memory allocated in the VRAM could not be freed.");
					return false;
				}

				//free volume part memory on gpu
				ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);

				if (!success) {
					this->lastErrorMessage = "Error: memory allocated in the VRAM could not be freed.";
					stopCudaThreads = true;
					return false;
				}

				currentSlice += sliceCnt;
			}

			std::cout << std::endl;
			std::cout << "GPU" << deviceId << " finished." << std::endl;

			emit(this->cudaThreadProgressUpdate(1, deviceId, true));

			return true;

		} catch (cv::Exception& e) {
			this->lastErrorMessage = std::string("An OpenCV error occured during the reconstruction: ") + e.what();
			return false;
		} catch (...) {
			this->lastErrorMessage = "An unidentified error occured during the CUDA reconstruction.";
			return false;
		}
	}

	bool CtVolume::launchCudaThreads() {
		this->stopCudaThreads = false;

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

		return result;
	}

	std::map<int, double> CtVolume::getGpuWeights(std::vector<int> const& devices) const {
		//get amount of multiprocessors per GPU
		std::map<int, int> multiprocessorCnt;
		size_t totalMultiprocessorsCnt = 0;
		std::map<int, int> bandwidth;
		size_t totalBandWidth = 0;
		for (int i = 0; i < devices.size(); ++i) {
			multiprocessorCnt[devices[i]] = ct::cuda::getMultiprocessorCnt(devices[i]);
			totalMultiprocessorsCnt += multiprocessorCnt[devices[i]];
			bandwidth[devices[i]] = ct::cuda::getMemoryBusWidth(devices[i]);
			totalBandWidth += bandwidth[devices[i]];
			std::cout << "GPU" << devices[i] << std::endl;
			cudaSetDevice(devices[i]);
			std::cout << "\tFree memory: " << double(ct::cuda::getFreeMemory()) / 1024 / 1024 / 1025 << " Gb" << std::endl;
			std::cout << "\tMultiprocessor count: " << multiprocessorCnt[devices[i]] << std::endl;
		}
		std::map<int, double> scalingFactors;
		double scalingFactorSum = 0;
		for (int i = 0; i < devices.size(); ++i) {
			double multiprocessorScalingFactor = (double(multiprocessorCnt[devices[i]]) / double(totalMultiprocessorsCnt));
			double busWidthScalingFactor = (double(bandwidth[devices[i]]) / double(totalBandWidth));
			scalingFactors[devices[i]] = multiprocessorScalingFactor*busWidthScalingFactor;
			scalingFactorSum += scalingFactors[devices[i]];
		}
		for (int i = 0; i < devices.size(); ++i) {
			scalingFactors[devices[i]] /= scalingFactorSum;
		}
		return scalingFactors;
	}

	size_t CtVolume::getMaxChunkSize(size_t fftSize) const {
		if (this->sinogram.size() < 1) return 0;
		long long freeMemory = ct::cuda::getFreeMemory();
		//spare some VRAM for other applications
		freeMemory -= this->gpuSpareMemory * 1024 * 1024;
		if (freeMemory < 0) return 0;

		size_t sliceSize = this->xMax * this->yMax * sizeof(float);
		if (sliceSize == 0) return 0;
		size_t sliceCnt = freeMemory / sliceSize;

		return sliceCnt;
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
				emit(reconstructionProgress(totalProgress, this->getVolumeCrossSection(this->crossSectionAxis, this->crossSectionIndex)));
			} else {
				emit(reconstructionProgress(totalProgress, cv::Mat()));
			}
		}
	}

}