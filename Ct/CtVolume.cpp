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

	size_t CtVolume::FftFilter::getWorkSizeEstimate(int width, int height) {
		size_t forwardFftSize, inverseFftSize;
		cufftEstimateMany(1, &width, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, height, &forwardFftSize);
		cufftEstimateMany(1, &width, NULL, 0, 0, NULL, 0, 0, CUFFT_C2R, height, &inverseFftSize);
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
			this->FCD = in.readLine().section('\t', 0, 0).toDouble(&success);
			totalSuccess = totalSuccess && success;
			this->baseIntensity = in.readLine().section('\t', 0, 0).toDouble(&success);
			totalSuccess = totalSuccess && success;
			//leave out one line
			in.readLine();
			//convert the distance
			this->FCD /= this->pixelSize;
			//convert uOffset
			this->uOffset /= this->pixelSize;
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
			unsigned int cnt = 0;
			unsigned int rows, cols;
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
		double radius = this->imageWidth / 2;
		radius = this->getReconstructionCylinderRadius();
		this->xSize = radius*2;
		this->ySize = radius*2;
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

	unsigned int CtVolume::getImageWidth() const {
		return this->imageWidth;
	}

	unsigned int CtVolume::getImageHeight() const {
		return this->imageHeight;
	}

	unsigned int CtVolume::getXSize() const {
		return this->xMax;
	}

	unsigned int CtVolume::getYSize() const {
		return this->yMax;
	}

	unsigned int CtVolume::getZSize() const {
		return this->zMax;
	}

	unsigned int CtVolume::getReconstructionCylinderRadius() const {
		double radius = this->imageWidth / 2;
		return std::sqrt((FCD*FCD * radius*radius) / (FCD*FCD + radius*radius));
	}

	float CtVolume::getUOffset() const {
		return this->uOffset;
	}

	float CtVolume::getPixelSize() const {
		return this->pixelSize;
	}

	float CtVolume::getFCD() const {
		return this->FCD;
	}

	float CtVolume::getBaseIntensity() const {
		return this->baseIntensity;
	}

	cv::Mat CtVolume::getVolumeCrossSection(Axis axis, unsigned int index) const {
		return this->volume.getVolumeCrossSection(axis, index, CoordinateSystemOrientation::LEFT_HANDED);
	}

	void CtVolume::setCrossSectionIndex(unsigned int index) {
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

	unsigned int CtVolume::getCrossSectionIndex() const {
		return this->crossSectionIndex;
	}

	unsigned int CtVolume::getCrossSectionSize() const {
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

	unsigned int CtVolume::getSkipProjections() const {
		return this->projectionStep - 1;
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

	double CtVolume::getMultiprocessorCoefficient() const {
		return this->multiprocessorCoefficient;
	}

	double CtVolume::getMemoryBandwidthCoefficient() const {
		return this->memoryBandwidthCoefficient;
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

	void CtVolume::setVolumeBounds(float xFrom, float xTo, float yFrom, float yTo, float zFrom, float zTo) {
		std::lock_guard<std::mutex> lock(this->exclusiveFunctionsMutex);
		if (xFrom == xTo || yFrom == yTo || zFrom == zTo) {
			std::cout << "ERROR: the lower and upper bounds must not be identical." << std::endl;
			return;
		}
		this->xFrom_float = std::max(0.0f, std::min(1.0f, xFrom));
		this->xTo_float = std::max(this->xFrom_float, std::min(1.0f, xTo));
		this->yFrom_float = std::max(0.0f, std::min(1.0f, yFrom));
		this->yTo_float = std::max(this->xFrom_float, std::min(1.0f, yTo));
		this->zFrom_float = std::max(0.0f, std::min(1.0f, zFrom));
		zTo_float = std::max(this->xFrom_float, std::min(1.0f, zTo));
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

	void CtVolume::setGpuCoefficients(double multiprocessorCoefficient, double memoryBandwidthCoefficient) {
		this->multiprocessorCoefficient = multiprocessorCoefficient;
		this->memoryBandwidthCoefficient = memoryBandwidthCoefficient;
	}

	void CtVolume::setFrequencyFilterType(FilterType filterType) {
		this->filterType = filterType;
	}

	void CtVolume::setSkipProjections(unsigned int value) {
		this->projectionStep = value + 1;
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

		QFileInfo fileInfo(filename);
		{
			//write information file
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
			out << "U resolution:\t\t" << this->imageWidth << endl;
			out << "V resolution:\t\t" << this->imageHeight << endl << endl;
			out << "[Reconstruction parameters]" << endl;
			out << "FCD:\t\t\t\t\t" << this->FCD << endl;
			out << "Pixel size:\t\t\t" << this->pixelSize << endl;
			out << "U offset:\t\t\t" << this->uOffset << endl;
			out << QString("X range relative:\t[%1 - %2]").arg(this->xFrom_float, 0, 'f', 3).arg(this->xTo_float, 0, 'f', 3) << endl;
			out << QString("Y range relative:\t[%1 - %2]").arg(this->yFrom_float, 0, 'f', 3).arg(this->yTo_float, 0, 'f', 3) << endl;
			out << QString("Z range relative:\t[%1 - %2]").arg(this->zFrom_float, 0, 'f', 3).arg(this->zTo_float, 0, 'f', 3) << endl;
			out << "X range:\t\t\t[" << this->xFrom << ".." << this->xTo << "]" << endl;
			out << "Y range:\t\t\t[" << this->yFrom << ".." << this->yTo << "]" << endl;
			out << "Z range:\t\t\t[" << this->zFrom << ".." << this->zTo << "]" << endl << endl;
			out << "[Volume dimensions]" << endl;
			out << "X size:\t\t\t\t" << this->xMax << endl;
			out << "Y size:\t\t\t\t" << this->yMax << endl;
			out << "Z size:\t\t\t\t" << this->zMax << endl << endl;
			out << "[Data format]" << endl;
			out << "Data type:\t\t\t32bit IEEE 754 float" << endl;
			if (byteOrder == QDataStream::LittleEndian) {
				out << "Byte order:\t\t\tLittle endian" << endl;
			} else {
				out << "Byte order:\t\t\tBig endian" << endl;
			}
			if (indexOrder == IndexOrder::Z_FASTEST) {
				out << "Index order:\t\tZ fastest";
			} else {
				out << "Index order:\t\tX fastest";
			}
			file.close();
		}
		{
		//write vgi file
			QString vgiFileName = QDir(fileInfo.path()).absoluteFilePath(fileInfo.baseName().append(".vgi"));
			QFile file(vgiFileName);
			if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
				std::cout << "Could not write the info file." << std::endl;
				if (this->emitSignals) emit(savingFinished(CompletionStatus::error("Could not write the vgi file.")));
				return;
			}
			QTextStream out(&file);
			//note: the new line at the end is important for VG Studio to be able to read the file
			QString contents = "{volume1}\n"
				"[representation]\n"
				"size = %1 %2 %3\n"
				"bitsperelement = 32\n"
				"datatype = float\n"
				"datarange = %4 %5\n"
				"[file1]\n"
				"FileFormat = raw\n"
				"Size = %1 %2 %3\n"
				"Name = ./%7\n"
				"bitsperelement = 32\n"
				"datatype = float\n"
				"datarange = %4 %5\n"
				"{volumeprimitive}\n"
				"[geometry]\n"
				"clipbox = 0 0 0 %1 %2 %3\n"
				"status = visible\n"
				"resolution = %6 %6 %6\n"
				"unit = mm\n"
				"[volume]\n"
				"volume = volume1\n";
			contents = contents.arg(this->xMax).arg(this->yMax).arg(this->zMax).arg(this->volume.min()).arg(this->volume.max()).arg(this->pixelSize).arg(fileInfo.fileName());
			out << contents;
			file.close();
		}

		//write binary file
		this->volume.saveToBinaryFile<float>(filename, indexOrder, QDataStream::SinglePrecision, byteOrder);

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
		QObject::connect(this, SIGNAL(cudaThreadProgressUpdate(double, int, bool)), this, SLOT(emitGlobalCudaProgress(double, int, bool)), Qt::QueuedConnection);
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

	cv::Mat CtVolume::prepareProjection(unsigned int index) const {
		cv::Mat image = this->sinogram[index].getImage();
		if (image.data) {
			convertTo32bit(image);
			this->preprocessImage(image);
		}
		return image;
	}

	void CtVolume::preprocessImage(cv::Mat& image) const {
		this->applyLogScaling(image);
		this->applyFeldkampWeight(image);
		applyFourierFilter(image, this->filterType);
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
				ptr[c] = ptr[c] * W(this->FCD, this->matToImageU(c), this->matToImageV(r));
			}
		}
	}

	inline float CtVolume::W(float D, float u, float v) {
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
		//scale FFT result
		//image *= 1.0 / static_cast<float>(image.cols);
	}

	void CtVolume::applyLogScaling(cv::Mat& image) const {
		image *= 1/this->baseIntensity;
		// -ln(x)
		cv::log(image, image);
		image *= -1;
	}

	float CtVolume::ramLakWindowFilter(float n, float N) {
		return n / N;
	}

	float CtVolume::sheppLoganWindowFilter(float n, float N) {
		if (n == 0) {
			return 0.0f;
		} else {
			float rl = ramLakWindowFilter(n, N);
			return (rl)* (sin(rl*0.5f*M_PI)) / (rl*0.5f*M_PI);
		}

	}

	float CtVolume::hannWindowFilter(float n, float N) {
		float rl = ramLakWindowFilter(n, N);
		return rl * 0.5f*(1.0f + cos(M_PI * rl));
	}

	void CtVolume::cudaPreprocessImage(cv::cuda::GpuMat& imageIn, cv::cuda::GpuMat& imageOut, cv::cuda::GpuMat& dftTmp, FftFilter& fftFilter, bool& success, cv::cuda::Stream& stream) const {
		success = true;
		bool successLocal;
		cudaStream_t cudaStream = cv::cuda::StreamAccessor::getStream(stream);
		//images must be scaled in case different depths are mixed (-> equal value range)
		float scalingFactor = 1.0;
		if (imageIn.depth() == CV_8U) {
			scalingFactor = 255.0;
		} else if (imageIn.depth() == CV_16U) {
			scalingFactor = 65535.0;
		}
		//convert to 32bit and normalise black
		imageIn.convertTo(imageOut, CV_32FC1, 1.0 / (scalingFactor*this->baseIntensity), stream);
		//logarithmic scale
		cv::cuda::log(imageOut, imageOut, stream);
		//multiply by -1
		cv::cuda::multiply(imageOut, -1, imageOut, 1.0, -1, stream);
		//apply the feldkamp weights
		ct::cuda::applyFeldkampWeightFiltering(imageOut, this->FCD, this->uPrecomputed, this->vPrecomputed, cudaStream, successLocal);
		success = success && successLocal;
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
		//scale FFT result
		//cv::cuda::multiply(imageOut, 1.0 / static_cast<float>(this->imageWidth), imageOut, 1.0, -1, stream);
	}

	bool CtVolume::reconstructionCore() {
		float imageLowerBoundU = this->matToImageU(0);
		float imageUpperBoundU = this->matToImageU(this->imageWidth - 1 - 0.1);
		//inversed because of inversed v axis in mat/image coordinate system
		float imageLowerBoundV = this->matToImageV(this->imageHeight - 1 - 0.1);
		float imageUpperBoundV = this->matToImageV(0);

		float volumeLowerBoundY = this->volumeToWorldY(0);
		float volumeUpperBoundY = this->volumeToWorldY(this->yMax);
		float volumeLowerBoundZ = this->volumeToWorldZ(0);
		float volumeUpperBoundZ = this->volumeToWorldZ(this->zMax);

		//copy some member variables to local variables, performance is better this way
		float FCD = this->FCD;
		float uOffset = this->uOffset;

		float radius = this->getReconstructionCylinderRadius();
		float radiusSquared = radius*radius;

		//for the preloading of the next projection
		std::future<cv::Mat> future;

		for (int projection = 0; projection < this->sinogram.size(); projection += this->projectionStep) {
			if (this->stopActiveProcess) {
				return false;
			}
			//output percentage
			float percentage = std::round((double)projection / (double)this->sinogram.size() * 100);
			std::cout << "\r" << "Backprojecting: " << percentage << "%";
			if (this->emitSignals) emit(reconstructionProgress(percentage, this->getVolumeCrossSection(this->crossSectionAxis, this->crossSectionIndex)));
			float angle_rad = (this->sinogram[projection].angle / 180.0) * M_PI;
			float sine = sin(angle_rad);
			float cosine = cos(angle_rad);
			//load the projection, the projection for the next iteration is already prepared in a background thread
			cv::Mat image;
			if (projection == 0) {
				image = this->prepareProjection(projection);
			} else {
				image = future.get();
			}
			if (projection + this->projectionStep < this->sinogram.size()) {
				future = std::async(std::launch::async, &CtVolume::prepareProjection, this, projection + this->projectionStep);
			}
			//check if the image is good
			if (!image.data) {
				this->lastErrorMessage = "The image " + this->sinogram[projection].imagePath.toStdString() + " could not be accessed. Maybe it doesn't exist or has an unsupported format.";
				return false;
			}
			//copy some member variables to local variables; performance is better this way
			float heightOffset = this->sinogram[projection].heightOffset;

			float* volumePtr;
#pragma omp parallel for private(volumePtr) schedule(dynamic)
			for (long xIndex = 0; xIndex < this->xMax; ++xIndex) {						//the loop has to be of an integer type for OpenMP to work
				float x = this->volumeToWorldX(xIndex);
				volumePtr = this->volume.slicePtr(xIndex);
				for (float y = volumeLowerBoundY; y < volumeUpperBoundY; ++y) {
					if ((x*x + y*y) >= radiusSquared) {
						volumePtr += this->zMax;
						continue;
					}
					//if the voxel is inside the reconstructable cylinder
					for (float z = volumeLowerBoundZ; z < volumeUpperBoundZ; ++z, ++volumePtr) {


						float reciprocalDistanceWeight, u, v;

						{
							float t = (-1)*x*sine + y*cosine;
							//correct the u-offset
							t += uOffset;
							float s = x*cosine + y*sine;
							reciprocalDistanceWeight = FCD / (FCD - s);
							u = t * reciprocalDistanceWeight;
							v = (z + heightOffset) * reciprocalDistanceWeight;
						}

						//check if it's inside the image (before the coordinate transformation)
						if (u >= imageLowerBoundU && u <= imageUpperBoundU && v >= imageLowerBoundV && v <= imageUpperBoundV) {

							u = this->imageToMatU(u);
							v = this->imageToMatV(v);

							float value;

							{
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

								//calculate weight
								float w = reciprocalDistanceWeight*reciprocalDistanceWeight;

								value = w * bilinearInterpolation(u - float(u0), v - float(v0), u0v0, u1v0, u0v1, u1v1);
							}

							(*volumePtr) += value;
						}
					}
				}
			}
		}
		std::cout << std::endl;
		return true;
	}

	inline float CtVolume::bilinearInterpolation(float u, float v, float u0v0, float u1v0, float u0v1, float u1v1) {
		//the two interpolations on the u axis
		double v0 = (1.0f - u)*u0v0 + u*u1v0;
		double v1 = (1.0f - u)*u0v1 + u*u1v1;
		//interpolation on the v axis between the two u-interpolated values
		return (1.0f - v)*v0 + v*v1;
	}

	bool CtVolume::cudaReconstructionCore(unsigned int threadZMin, unsigned int threadZMax, int deviceId) {

		try {

			cudaSetDevice(deviceId);
			//for cuda error handling
			bool success;

			//precomputing some values
			double imageLowerBoundU = this->matToImageU(0);
			double imageUpperBoundU = this->matToImageU(this->imageWidth - 1 - 0.1);
			//inversed because of inversed v axis in mat/image coordinate system
			double imageLowerBoundV = this->matToImageV(this->imageHeight - 1 - 0.1);
			double imageUpperBoundV = this->matToImageV(0);
			double radius = this->getReconstructionCylinderRadius();
			double radiusSquared = radius*radius;

			const unsigned int progressUpdateRate = std::max(this->sinogram.size() / 100 * this->getActiveCudaDevices().size(), static_cast<size_t>(1));

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

			unsigned int sliceCnt = getMaxChunkSize();
			unsigned int currentSlice = threadZMin;
			//slice count equivalent to 150mb memory usage
			unsigned int decreaseSliceStep = static_cast<unsigned int>(std::max(std::ceil(150.0L * 1024.0L * 1024.0L / static_cast<long double>(this->xMax*this->yMax*sizeof(float))), 1.0L));

			if (sliceCnt < 1) {
				//too little memory
				this->lastErrorMessage = "The VRAM of one of the used GPUs is insufficient to run a reconstruction.";
				//stop also the other cuda threads
				stopCudaThreads = true;
				return false;
			}

			sliceCnt = std::min(sliceCnt, threadZMax - threadZMin);

			std::cout << std::endl;
			//allocate volume part memory on gpu
			cudaPitchedPtr gpuVolumePtr;
			do {
				gpuVolumePtr = ct::cuda::create3dVolumeOnGPU(this->xMax, this->yMax, sliceCnt, success, false);
				if (!success) {
					std::cout << "GPU" << deviceId << " tries allocating " << sliceCnt << " slices. FAIL" << std::endl;;
					if (decreaseSliceStep < sliceCnt && decreaseSliceStep > 0) {
						sliceCnt -= decreaseSliceStep;
						//this resets sticky errors
						cudaGetLastError();
					} else {
						break;
					}
				} else {
					std::cout << "GPU" << deviceId << " tries allocating " << sliceCnt << " slices. SUCCESS" << std::endl;
				}
			} while (!success && cudaGetLastError() == cudaSuccess);

			if (!success) {
				this->lastErrorMessage = "An error occured during allocation of memory for the volume in the VRAM. Maybe the amount of free VRAM was insufficient. You can try changing the GPU spare memory setting.";
				stopCudaThreads = true;
				return false;
			}

			while (currentSlice < threadZMax) {

				unsigned int lastSlice = std::min(currentSlice + sliceCnt, threadZMax);
				unsigned int const xDimension = this->xMax;
				unsigned int const yDimension = this->yMax;
				unsigned int zDimension = lastSlice - currentSlice;

				std::cout << std::endl << "GPU" << deviceId << " processing [" << currentSlice << ".." << lastSlice << ")" << std::endl;

				ct::cuda::setToZero(gpuVolumePtr, this->xMax, this->yMax, sliceCnt, success);

				if (!success) {
					this->lastErrorMessage = "An error occured during initialisation of the volume in the VRAM.";
					ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
					stopCudaThreads = true;
					return false;
				}

				int current = 0;
				int other = 1;

				for (int projection = 0; projection < this->sinogram.size(); projection += this->projectionStep) {

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
						double chunkFinished = static_cast<double>(currentSlice - threadZMin)*static_cast<double>(this->xMax)*static_cast<double>(this->yMax);
						double currentChunk = static_cast<double>(zDimension)*static_cast<double>(this->xMax)*static_cast<double>(this->yMax) * (double(projection) / double(this->sinogram.size()));
						double percentage = (chunkFinished + currentChunk) / (static_cast<double>(threadZMax - threadZMin)*static_cast<double>(this->xMax)*static_cast<double>(this->yMax));
						emit(this->cudaThreadProgressUpdate(percentage, deviceId, (projection == 0)));
					}

					std::swap(current, other);

					double angle_rad = (this->sinogram[projection].angle / 180.0) * M_PI;
					double sine = sin(angle_rad);
					double cosine = cos(angle_rad);

					//read next image from disk
					image = this->sinogram[projection].getImage();
					if (!image.data) {
						this->lastErrorMessage = "The image " + this->sinogram[projection].imagePath.toStdString() + " could not be accessed. Maybe it doesn't exist or has an unsupported format.";
						stopCudaThreads = true;
						ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
						if (!success) this->lastErrorMessage += std::string(" Some memory allocated in the VRAM could not be freed.");
						return false;
					}
					try {
						//wait for this stream to finish the last reconstruction (otherwise we overwrite the image while it's still in use)
						//stream[current].waitForCompletion();	<- unnecessary
						//then copy image to page locked memory and upload it to the gpu
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

					//wait for other stream to finish its last reconstruction
					stream[other].waitForCompletion();

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
												  this->FCD,
												  imageLowerBoundU,
												  imageUpperBoundU,
												  imageLowerBoundV,
												  imageUpperBoundV,
												  this->xPrecomputed,
												  this->yPrecomputed,
												  this->zPrecomputed,
												  this->uPrecomputed,
												  this->vPrecomputed,
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

				//make sure both streams are ready
				stream[0].waitForCompletion();
				stream[1].waitForCompletion();

				//donload the reconstructed volume part
				ct::cuda::download3dVolume(gpuVolumePtr, this->volume.slicePtr(currentSlice), xDimension, yDimension, zDimension, success);

				if (!success) {
					this->lastErrorMessage = "An error occured during download of a reconstructed volume part from the VRAM.";
					stopCudaThreads = true;
					ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);
					if (!success) this->lastErrorMessage += std::string(" Some memory allocated in the VRAM could not be freed.");
					return false;
				}

				if (!success) {
					this->lastErrorMessage = "Error: memory allocated in the VRAM could not be freed.";
					stopCudaThreads = true;
					return false;
				}

				currentSlice += sliceCnt;
			}

			//free volume part memory on gpu
			ct::cuda::delete3dVolumeOnGPU(gpuVolumePtr, success);

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

		//more L1 cache; we don't need shared memory
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		this->cudaGpuWeights = this->getGpuWeights(this->activeCudaDevices);

		//clear progress
		this->cudaThreadProgress.clear();
		for (int i = 0; i < this->activeCudaDevices.size(); ++i) {
			this->cudaThreadProgress.insert(std::make_pair(activeCudaDevices[i], 0));
		}

		//create vector to store threads
		std::vector<std::future<bool>> threads(this->activeCudaDevices.size());
		//launch one thread for each part of the volume (weighted by the amount of multiprocessors)
		unsigned int currentSlice = 0;
		for (int i = 0; i < this->activeCudaDevices.size(); ++i) {
			unsigned int sliceCnt = std::round(this->cudaGpuWeights[this->activeCudaDevices[i]] * double(zMax));
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
		unsigned int totalMultiprocessorsCnt = 0;
		std::map<int, int> bandwidth;
		size_t totalBandWidth = 0;
		for (int i = 0; i < devices.size(); ++i) {
			multiprocessorCnt[devices[i]] = ct::cuda::getMultiprocessorCnt(devices[i]);
			totalMultiprocessorsCnt += multiprocessorCnt[devices[i]];
			bandwidth[devices[i]] = ct::cuda::getMemoryBusWidth(devices[i])*ct::cuda::getMemoryClockRate(devices[i]);
			totalBandWidth += bandwidth[devices[i]];
		}
		std::map<int, double> scalingFactors;
		double scalingFactorSum = 0;
		for (int i = 0; i < devices.size(); ++i) {
			double multiprocessorScalingFactor = (double(multiprocessorCnt[devices[i]]) / double(totalMultiprocessorsCnt));
			double bandwidthScalingFactor = (double(bandwidth[devices[i]]) / double(totalBandWidth));
			scalingFactors[devices[i]] = std::pow(multiprocessorScalingFactor, this->multiprocessorCoefficient)*std::pow(bandwidthScalingFactor, this->memoryBandwidthCoefficient);
			scalingFactorSum += scalingFactors[devices[i]];
		}
		for (int i = 0; i < devices.size(); ++i) {
			scalingFactors[devices[i]] /= scalingFactorSum;
			std::cout << "GPU" << devices[i] << std::endl;
			cudaSetDevice(devices[i]);
			std::cout << "\tFree memory: " << double(ct::cuda::getFreeMemory()) / 1024.0 / 1024.0 / 1025.0 << " Gb" << std::endl;
			std::cout << "\tMultiprocessor count: " << multiprocessorCnt[devices[i]] << std::endl;
			std::cout << "\tPeak memory bandwidth: " << (double(bandwidth[devices[i]]) * 2.0) / 8000000.0 << " Gb/s" << std::endl;
			std::cout << "\tGPU weight: " << scalingFactors[devices[i]] << std::endl;
		}
		return scalingFactors;
	}

	unsigned int CtVolume::getMaxChunkSize() const {
		if (this->sinogram.size() < 1) return 0;
		long long freeMemory = ct::cuda::getFreeMemory();
		//spare some VRAM for other applications
		freeMemory -= this->gpuSpareMemory * 1024 * 1024;
		if (freeMemory < 0) return 0;

		size_t sliceSize = this->xMax * this->yMax * sizeof(float);
		if (sliceSize == 0) return 0;
		unsigned int sliceCnt = freeMemory / sliceSize;

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
		this->xPrecomputed = (float(this->xSize) / 2.0) - float(this->xFrom);
		this->yPrecomputed = (float(this->ySize) / 2.0) - float(this->yFrom);
		this->zPrecomputed = (float(this->zSize) / 2.0) - float(this->zFrom);
		this->uPrecomputed = float(this->imageWidth) / 2.0;
		this->vPrecomputed = float(this->imageHeight) / 2.0;
	}

	inline float CtVolume::worldToVolumeX(float xCoord) const {
		return xCoord + this->xPrecomputed;
	}

	inline float CtVolume::worldToVolumeY(float yCoord) const {
		return yCoord + this->yPrecomputed;
	}

	inline float CtVolume::worldToVolumeZ(float zCoord) const {
		return zCoord + this->zPrecomputed;
	}

	inline float CtVolume::volumeToWorldX(float xCoord) const {
		return xCoord - this->xPrecomputed;
	}

	inline float CtVolume::volumeToWorldY(float yCoord) const {
		return yCoord - this->yPrecomputed;
	}

	inline float CtVolume::volumeToWorldZ(float zCoord) const {
		return zCoord - this->zPrecomputed;
	}

	inline float CtVolume::imageToMatU(float uCoord)const {
		return uCoord + this->uPrecomputed;
	}

	inline float CtVolume::imageToMatV(float vCoord)const {
		//factor -1 because of different z-axis direction
		return (-1)*vCoord + this->vPrecomputed;
	}

	inline float CtVolume::matToImageU(float uCoord)const {
		return uCoord - this->uPrecomputed;
	}

	inline float CtVolume::matToImageV(float vCoord)const {
		//factor -1 because of different z-axis direction
		return (-1)*vCoord + this->vPrecomputed;
	}

	//============================================== PRIVATE SLOTS ==============================================\\

	void CtVolume::emitGlobalCudaProgress(double percentage, int deviceId, bool emitCrossSection) {
		this->cudaThreadProgress[deviceId] = percentage;
		double totalProgress = 0;
		for (int i = 0; i < this->activeCudaDevices.size(); ++i) {
			totalProgress += this->cudaThreadProgress[this->activeCudaDevices[i]] * this->cudaGpuWeights[this->activeCudaDevices[i]];
		}
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