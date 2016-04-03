//note: this class is header-only

#ifndef CT_VOLUME
#define CT_VOLUME

#include <atomic>

//OpenCV
#include <opencv2/core/core.hpp>				//core functionality of OpenCV

//Qt
#include <QtCore/QtCore>

#include "Types.h"

namespace ct {

	enum class Axis {
		X,
		Y,
		Z
	};

	enum class CoordinateSystemOrientation {
		LEFT_HANDED,
		RIGHT_HANDED
	};

	enum class IndexOrder {
		X_FASTEST,
		Z_FASTEST
	};

	//base class for signals (signals do not work in template class)
	class AbstractVolume : public QObject {
		Q_OBJECT
	public:
		virtual ~AbstractVolume() = default;
		virtual void clear() = 0;
		virtual bool saveToBinaryFile(QString const& filename,							//saves the volume to a binary file with the given filename
									  IndexOrder indexOrder = IndexOrder::Z_FASTEST,
									  QDataStream::FloatingPointPrecision floatingPointPrecision = QDataStream::SinglePrecision,
									  QDataStream::ByteOrder byteOrder = QDataStream::LittleEndian) const = 0;
		virtual cv::Mat getVolumeCrossSection(Axis axis,								//returns a cross section through the volume as image
											  size_t index,
											  CoordinateSystemOrientation type) const = 0;
		virtual size_t getSizeAlongDimension(Axis axis) const = 0;						//returns the size along the axis axis
		virtual void stop() = 0;														//stops the saving function
		//getters
		virtual bool getEmitSignals() const = 0;
		virtual size_t xSize() const = 0;
		virtual size_t ySize() const = 0;
		virtual size_t zSize() const = 0;
		virtual size_t sliceCnt() const = 0;
		virtual size_t sliceSize() const = 0;
		virtual size_t rowSize() const = 0;
		virtual double atRelative(size_t x, size_t y, size_t z) const = 0;
		virtual float minFloat() const = 0;
		virtual float maxFloat() const = 0;
		//setters
		virtual void setMemoryLayout(IndexOrder indexOrder) = 0;
		virtual void setEmitSignals(bool value) = 0;
	signals:
		void savingProgress(double percentage) const;
		void savingFinished(CompletionStatus status = CompletionStatus::success()) const;
		void loadingProgress(double percentage) const;
		void loadingFinished(CompletionStatus status = CompletionStatus::success()) const;
	};

	template <typename T>
	class Volume : public AbstractVolume {
	public:
		Volume() = default;
		Volume(size_t xSize, size_t ySize, size_t zSize, T defaultValue = 0);
		~Volume();
		Volume& operator=(Volume const& other) = delete;
		void reinitialise(size_t xSize,											//resizes the volume to the given dimensions and sets all elements to the given value
						  size_t ySize, 
						  size_t zSize, 
						  T defaultValue = 0);
		void clear();
		template <typename U>
		bool loadFromBinaryFile(QString const& filename,						//reads a volume from a binary file
								size_t xSize,
								size_t ySize,
								size_t zSize,
								IndexOrder indexOrder = IndexOrder::Z_FASTEST,
								QDataStream::FloatingPointPrecision floatingPointPrecision = QDataStream::SinglePrecision,
								QDataStream::ByteOrder byteOrder = QDataStream::LittleEndian);
		bool saveToBinaryFile(QString const& filename,							//saves the volume to a binary file with the given filename
							  IndexOrder indexOrder = IndexOrder::Z_FASTEST,
							  QDataStream::FloatingPointPrecision floatingPointPrecision = QDataStream::SinglePrecision, 
							  QDataStream::ByteOrder byteOrder = QDataStream::LittleEndian) const;				
		cv::Mat getVolumeCrossSection(Axis axis,								//returns a cross section through the volume as image
									  size_t index, 
									  CoordinateSystemOrientation type) const;			
		size_t getSizeAlongDimension(Axis axis) const;							//returns the size along the axis axis
		void stop();															//stops the saving function
		//getters
		bool getEmitSignals() const;
		size_t xSize() const;
		size_t ySize() const;
		size_t zSize() const;
		size_t sliceCnt() const;
		size_t sliceSize() const;
		size_t rowSize() const;
		T& at(size_t x, size_t y, size_t z);
		T const& at(size_t x, size_t y, size_t z) const;
		double atRelative(size_t x, size_t y, size_t z) const;
		T min() const;
		T max() const;
		float minFloat() const;
		float maxFloat() const;
		T* data();
		T const* data() const;
		T* slicePtr(size_t sliceIndex);
		T const* slicePtr(size_t sliceIndex) const;
		T* rowPtr(size_t sliceIndex, size_t rowIndex);
		T const* rowPtr(size_t sliceIndex, size_t rowIndex) const;
		//setters
		void setMemoryLayout(IndexOrder indexOrder);
		void setEmitSignals(bool value);
	private:
		//functions
		void calculateMinMax() const;

		//variables
		T* volume = nullptr;
		mutable T minValue = 0;
		mutable T maxValue = 0;
		mutable bool minMaxCalculated = false;
		size_t xMax = 0, yMax = 0, zMax = 0, slicePitchXFastest = 0, slicePitchZFastest = 0;
		IndexOrder mode = IndexOrder::Z_FASTEST;
		bool emitSignals = true;												//if true the object emits qt signals in certain functions
		mutable std::atomic<bool> stopActiveProcess{ false };
	};

	//=========================================== IMPLEMENTATION ===========================================\\

	template <typename T>
	Volume<T>::Volume(size_t xSize, size_t ySize, size_t zSize, T defaultValue) {
		this->reinitialise(xSize, ySize, zSize, defaultValue);
	}

	template<typename T>
	inline Volume<T>::~Volume() {
		if (this->volume != nullptr) {
			delete[] this->volume;
		}
	}

	template <typename T>
	void Volume<T>::reinitialise(size_t xSize, size_t ySize, size_t zSize, T defaultValue) {
		this->clear();
		this->volume = new T[xSize*ySize*zSize];
		std::fill(this->volume, this->volume + xSize*ySize*zSize, defaultValue);
		this->xMax = xSize;
		this->yMax = ySize;
		this->zMax = zSize;
		this->slicePitchXFastest = ySize * xSize;
		this->slicePitchZFastest = ySize * zSize;
	}

	template<typename T>
	void Volume<T>::clear() {
		if (this->volume != nullptr) {
			delete[] this->volume;
			this->volume = nullptr;
			this->xMax = 0;
			this->yMax = 0;
			this->zMax = 0;
			this->minMaxCalculated = false;
		}
	}

	template<typename T>
	void Volume<T>::setMemoryLayout(IndexOrder indexOrder) {
		this->mode = indexOrder;
	}

	template<typename T>
	void Volume<T>::setEmitSignals(bool value) {
		this->emitSignals = value;
	}

	template<typename T>
	void Volume<T>::calculateMinMax() const {
		T threadMin = std::numeric_limits<T>::max();
		T threadMax = std::numeric_limits<T>::lowest();
		T min = std::numeric_limits<T>::max();
		T max = std::numeric_limits<T>::lowest();
		T const* ptr;
#pragma omp parallel private(ptr, threadMin, threadMax)
		{
#pragma omp for
			for (int slice = 0; slice < this->sliceCnt(); ++slice) {
				ptr = this->slicePtr(slice);
				for (int i = 0; i < this->sliceSize(); ++i, ++ptr) {
					if (*ptr < threadMin) threadMin = *ptr;
					if (*ptr > threadMax) threadMax = *ptr;
				}
			}
#pragma omp critical(sync)
			{
				if (threadMin < min) min = threadMin;
				if (threadMax > max) max = threadMax;
			}
		}
		this->minValue = min;
		this->maxValue = max;
		this->minMaxCalculated = true;
	}

	template<typename T>
	bool Volume<T>::getEmitSignals() const {
		return this->emitSignals;
	}

	template<typename T>
	size_t Volume<T>::xSize() const {
		return this->xMax;
	}

	template<typename T>
	size_t Volume<T>::ySize() const {
		return this->yMax;
	}

	template<typename T>
	size_t Volume<T>::zSize() const {
		return this->zMax;
	}

	template<typename T>
	inline size_t Volume<T>::sliceCnt() const {
		if (this->mode == IndexOrder::X_FASTEST) return this->zSize();
		return this->xSize();
	}

	template<typename T>
	inline size_t Volume<T>::sliceSize() const {
		if (this->mode == IndexOrder::X_FASTEST) return this->ySize() * this->xSize();
		return this->ySize() * this->zSize();
	}

	template<typename T>
	inline size_t Volume<T>::rowSize() const {
		if (this->mode == IndexOrder::X_FASTEST) return this->xSize();
		return this->zSize();
	}

	template<typename T>
	inline T& Volume<T>::at(size_t x, size_t y, size_t z) {
		if (x >= this->xMax || y >= this->yMax || z >= this->zMax) throw std::out_of_range("Volume index out of bounds");
		if (this->mode == IndexOrder::Z_FASTEST) return this->volume[x * this->slicePitchZFastest + y*this->zMax + z];
		return this->volume[z * this->slicePitchXFastest + y*this->xMax + x];
	}

	template<typename T>
	inline T const& Volume<T>::at(size_t x, size_t y, size_t z) const {
		if (x >= this->xMax || y >= this->yMax || z >= this->zMax) throw std::out_of_range("Volume index out of bounds");
		if(this->mode == IndexOrder::Z_FASTEST) return this->volume[x * this->slicePitchZFastest + y*this->zMax + z];
		return this->volume[z * this->slicePitchXFastest + y*this->xMax + x];
	}

	template<typename T>
	inline double Volume<T>::atRelative(size_t x, size_t y, size_t z) const {
		double min = static_cast<double>(this->min());
		double span = static_cast<double>(this->max()) - min;
		return (static_cast<double>(this->at(x, y, z)) - min) / span;
	}

	template<typename T>
	inline T Volume<T>::min() const {
		if (!this->minMaxCalculated) this->calculateMinMax();
		return this->minValue;
	}

	template<typename T>
	inline T Volume<T>::max() const {
		if (!this->minMaxCalculated) this->calculateMinMax();
		return this->maxValue;
	}

	template<typename T>
	inline float Volume<T>::minFloat() const {
		return static_cast<float>(this->min());
	}

	template<typename T>
	inline float Volume<T>::maxFloat() const {
		return static_cast<float>(this->max());
	}

	template<typename T>
	T* Volume<T>::data() {
		return this->volume;
	}

	template<typename T>
	inline T const* Volume<T>::data() const {
		return this->volume;
	}

	template<typename T>
	inline T* Volume<T>::slicePtr(size_t outerIndex) {
		if (this->mode == IndexOrder::Z_FASTEST) return &this->volume[outerIndex * this->slicePitchZFastest];
		return &this->volume[outerIndex * this->slicePitchXFastest];
	}

	template<typename T>
	inline T const* Volume<T>::slicePtr(size_t outerIndex) const {
		if (this->mode == IndexOrder::Z_FASTEST) return &this->volume[outerIndex * this->slicePitchZFastest];
		return &this->volume[outerIndex * this->slicePitchXFastest];
	}

	template<typename T>
	inline T* Volume<T>::rowPtr(size_t outerIndex, size_t innerIndex) {
		if (this->mode == IndexOrder::Z_FASTEST) return &this->volume[outerIndex * this->slicePitchZFastest + innerIndex*this->zMax];
		return &this->volume[outerIndex * this->slicePitchXFastest + innerIndex*this->xMax];
	}

	template<typename T>
	inline T const* Volume<T>::rowPtr(size_t outerIndex, size_t innerIndex) const {
		if (this->mode == IndexOrder::Z_FASTEST) return &this->volume[outerIndex * this->slicePitchZFastest + innerIndex*this->zMax];
		return &this->volume[outerIndex * this->slicePitchXFastest + innerIndex*this->xMax];
	}

	template <typename T>
	template <typename U>
	bool Volume<T>::loadFromBinaryFile(QString const& filename, size_t xSize, size_t ySize, size_t zSize, IndexOrder indexOrder, QDataStream::FloatingPointPrecision floatingPointPrecision, QDataStream::ByteOrder byteOrder) {
		this->stopActiveProcess = false;
		size_t voxelSize = 0;
		if (std::is_floating_point<U>::value) {
			if (floatingPointPrecision == QDataStream::SinglePrecision) {
				voxelSize = 4; //32 bit
			} else {
				voxelSize = 8; //64 bit
			}
		} else {
			voxelSize = sizeof(U);
		}
		size_t totalFileSize = xSize * ySize * zSize * voxelSize;
		size_t actualFileSize = QFileInfo(filename).size();
		QFile file(filename);
		if (!file.open(QIODevice::ReadOnly)) {
			std::cout << "Could not open the file. Maybe your path does not exist." << std::endl;
			if (this->emitSignals) emit(loadingFinished(CompletionStatus::error("Could not open the file. Maybe your path does not exist.")));
			return false;
		}
		if (actualFileSize != totalFileSize) {
			QString message = QString("The size of the file does not fit the given parameters. Expected filesize: %1 Actual filesize: %2").arg(totalFileSize).arg(actualFileSize);
			std::cout << message.toStdString() << std::endl;
			if (this->emitSignals) emit(loadingFinished(CompletionStatus::error(message)));
			return false;
		}
		this->reinitialise(xSize, ySize, zSize);
		QDataStream in(&file);
		in.setFloatingPointPrecision(floatingPointPrecision);
		in.setByteOrder(byteOrder);
		//iterate through the volume
		int x, z;
		int xUpperBound = this->xSize(), zUpperBound = this->zSize();
		int* innerIndex, *innerMax, *outerIndex, *outerMax;
		if (indexOrder == IndexOrder::X_FASTEST) {
			innerIndex = &x, outerIndex = &z;
			innerMax = &xUpperBound, outerMax = &zUpperBound;
		} else {
			innerIndex = &z, outerIndex = &x;
			innerMax = &zUpperBound, outerMax = &xUpperBound;
		}
		T min = std::numeric_limits<T>::max();
		T max = std::numeric_limits<T>::lowest();
		U tmp;
		T converted;
		if (this->mode == indexOrder) {
			T* volumePtr = this->volume;
			size_t size = this->xMax*this->yMax*this->zMax;
			for (int i = 0; i < size; ++i, ++volumePtr) {
				if (i % 100000 == 0) {
					if (this->stopActiveProcess) {
						this->clear();
						std::cout << "User interrupted. Stopping." << std::endl;
						if (this->emitSignals) emit(loadingFinished(CompletionStatus::interrupted()));
						return false;
					}
					double percentage = std::round(double(i) / double(size) * 100);
					if(this->emitSignals) emit(loadingProgress(percentage));
				}
				//load one U of data
				in >> tmp;
				if (in.status() != QDataStream::Ok) {
					if (in.status() == QDataStream::ReadCorruptData) {
						std::cout << "An error occured while writing from the disk. The data seems to be corrupted." << std::endl;
						if (this->emitSignals) emit(savingFinished(CompletionStatus::error("An error occured while reading from the disk. The data seems to be corrupted.")));
						return false;
					} else {
						std::cout << "An error occured while writing from the disk." << std::endl;
						if (this->emitSignals) emit(savingFinished(CompletionStatus::error("An error occured while writing from the disk.")));
						return false;
					}
				}
				converted = static_cast<T>(tmp);
				if (converted < min) min = converted;
				if (converted > max) max = converted;
				(*volumePtr) = converted;
			}
		} else {
			for (*outerIndex = 0; *outerIndex < *outerMax; ++(*outerIndex)) {
				if (this->stopActiveProcess) {
					this->clear();
					std::cout << "User interrupted. Stopping." << std::endl;
					if (this->emitSignals) emit(loadingFinished(CompletionStatus::interrupted()));
					return false;
				}
				double percentage = std::round(double(*outerIndex) / double(*outerMax) * 100);
				if (this->emitSignals) emit(loadingProgress(percentage));
				for (int y = 0; y < this->ySize(); ++y) {
					for (*innerIndex = 0; *innerIndex < *innerMax; ++(*innerIndex)) {
						//load one U of data
						in >> tmp;
						if (in.status() != QDataStream::Ok) {
							if (in.status() == QDataStream::ReadCorruptData) {
								std::cout << "An error occured while writing from the disk. The data seems to be corrupted." << std::endl;
								if (this->emitSignals) emit(savingFinished(CompletionStatus::error("An error occured while reading from the disk. The data seems to be corrupted.")));
								return false;
							} else {
								std::cout << "An error occured while writing from the disk." << std::endl;
								if (this->emitSignals) emit(savingFinished(CompletionStatus::error("An error occured while writing from the disk.")));
								return false;
							}
						}
						converted = static_cast<T>(tmp);
						if (converted < min) min = converted;
						if (converted > max) max = converted;
						this->at(x, y, z) = converted;
					}
				}
			}
		}
		file.close();
		this->minValue = min;
		this->maxValue = max;
		this->minMaxCalculated = true;
		if (this->emitSignals) emit(loadingFinished());
		return true;
	}

	template <typename T>
	bool Volume<T>::saveToBinaryFile(QString const& filename, IndexOrder indexOrder, QDataStream::FloatingPointPrecision floatingPointPrecision, QDataStream::ByteOrder byteOrder) const {
		this->stopActiveProcess = false;
		if (this->xSize() > 0 && this->ySize() > 0 && this->zSize() > 0) {
			{
				//write binary file
				QFile file(filename);
				if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
					std::cout << "Could not open the file. Maybe your path does not exist. No files were written." << std::endl;
					if(this->emitSignals) emit(savingFinished(CompletionStatus::error("Could not open the file. Maybe your path does not exist. No files were written.")));
					return false;
				}
				QDataStream out(&file);
				out.setFloatingPointPrecision(floatingPointPrecision);
				out.setByteOrder(byteOrder);
				//iterate through the volume
				int x, z;
				int xUpperBound = this->xSize(), zUpperBound = this->zSize();
				int* innerIndex, *innerMax, *outerIndex, *outerMax;
				if (indexOrder == IndexOrder::X_FASTEST) {
					innerIndex = &x, outerIndex = &z;
					innerMax = &xUpperBound, outerMax = &zUpperBound;
				} else {
					innerIndex = &z, outerIndex = &x;
					innerMax = &zUpperBound, outerMax = &xUpperBound;
				}
				if (this->mode == indexOrder) {
					T* volumePtr = this->volume;
					size_t size = this->xMax*this->yMax*this->zMax;
					for (int i = 0; i < size; ++i, ++volumePtr) {
						if (i % 100000 == 0) {
							if (this->stopActiveProcess) {
								std::cout << "User interrupted. Stopping." << std::endl;
								if (this->emitSignals) emit(savingFinished(CompletionStatus::interrupted()));
								return false;
							}
							double percentage = std::round(double(i) / double(size) * 100);
							if(this->emitSignals && !this->stopActiveProcess) emit(savingProgress(percentage));
						}
						//save one T of data
						out << (*volumePtr);
						if (out.status() != QDataStream::Ok) {
							if (out.status() == QDataStream::WriteFailed) {
								std::cout << "An error occured while writing to the disk. Maybe there is not enough free disk space." << std::endl;
								if (this->emitSignals) emit(savingFinished(CompletionStatus::error("An error occured while writing to the disk. Maybe there is not enough free disk space.")));
								return false;
							} else {
								std::cout << "An error occured while writing to the disk." << std::endl;
								if (this->emitSignals) emit(savingFinished(CompletionStatus::error("An error occured while writing to the disk.")));
								return false;
							}
						}
					}
				} else {
					for (*outerIndex = 0; *outerIndex < *outerMax; ++(*outerIndex)) {
						if (this->stopActiveProcess) {
							std::cout << "User interrupted. Stopping." << std::endl;
							if (this->emitSignals) emit(savingFinished(CompletionStatus::interrupted()));
							return false;
						}
						double percentage = std::round(double(*outerIndex) / double(*outerMax) * 100);
						if (this->emitSignals && !this->stopActiveProcess) emit(savingProgress(percentage));
						for (int y = 0; y < this->ySize(); ++y) {
							for (*innerIndex = 0; *innerIndex < *innerMax; ++(*innerIndex)) {
								//save one T of data
								out << this->at(x, y, z);
								if (out.status() != QDataStream::Ok) {
									if (out.status() == QDataStream::WriteFailed) {
										std::cout << "An error occured while writing to the disk. Maybe there is not enough free disk space." << std::endl;
										if (this->emitSignals) emit(savingFinished(CompletionStatus::error("An error occured while writing to the disk. Maybe there is not enough free disk space.")));
										return false;
									} else {
										std::cout << "An error occured while writing to the disk." << std::endl;
										if (this->emitSignals) emit(savingFinished(CompletionStatus::error("An error occured while writing to the disk.")));
										return false;
									}
								}
							}
						}
					}
				}

				file.close();
			}
		} else {
			std::cout << "Did not save the volume, because it appears to be empty." << std::endl;
			if (this->emitSignals) emit(savingFinished(CompletionStatus::error("Did not save the volume, because it appears to be empty.")));
			return false;
		}
		std::cout << "Volume successfully saved." << std::endl;
		if (this->emitSignals) emit(savingFinished());
		return true;
	}

	template<typename T>
	cv::Mat Volume<T>::getVolumeCrossSection(Axis axis, size_t index, CoordinateSystemOrientation type) const {
		if (this->xSize() == 0) return cv::Mat();
		if (index >= 0 && ((axis == Axis::X && index < this->xSize()) || (axis == Axis::Y && index < this->ySize()) || (axis == Axis::Z && index < this->zSize()))) {
			if (this->xSize() > 0 && this->ySize() > 0 && this->zSize() > 0) {
				size_t uSize;
				size_t vSize;
				switch (axis) {
					case Axis::X:
						uSize = this->ySize();
						vSize = this->zSize();
						break;
					case Axis::Y:
						uSize = this->xSize();
						vSize = this->zSize();
						break;
					case Axis::Z:
						uSize = this->ySize();
						vSize = this->xSize();
						break;
				}

				cv::Mat result(static_cast<int>(vSize), static_cast<int>(uSize), CV_32FC1);
				std::function<void(int, int, float*)> const setPixel = [&]()->std::function<void(int, int, float*)> {
					switch (axis) {
						case Axis::X:
							if (type == CoordinateSystemOrientation::LEFT_HANDED) {
								return [&](int row, int column, float* ptr) {
									ptr[column] = static_cast<float>(this->at(index, column, result.rows - 1 - row));
								};
							} else {
								return [&](int row, int column, float* ptr) {
									ptr[column] = static_cast<float>(this->at(index, result.cols - 1 - column, result.rows - 1 - row));
								};
							}
						case Axis::Y:
							if (type == CoordinateSystemOrientation::LEFT_HANDED) {
								return [&](int row, int column, float* ptr) {
									ptr[column] = static_cast<float>(this->at(result.cols - 1 - column, index, result.rows - 1 - row));
								};
							} else {
								return [&](int row, int column, float* ptr) {
									ptr[column] = static_cast<float>(this->at(column, index, result.rows - 1 - row));
								};
							}
						case Axis::Z:
							if (type == CoordinateSystemOrientation::LEFT_HANDED) {
								return [&](int row, int column, float* ptr) {
									ptr[column] = static_cast<float>(this->at(row, column, index));
								};
							} else {
								return [&](int row, int column, float* ptr) {
									ptr[column] = static_cast<float>(this->at(row, result.cols - 1 - column, index));
								};
							}
					}
				}();

				float* ptr;
#pragma omp parallel for private(ptr)
				for (int row = 0; row < result.rows; ++row) {
					ptr = result.ptr<float>(row);
					for (int column = 0; column < result.cols; ++column) {
						setPixel(row, column, ptr);
					}
				}
				return result;
			}
			return cv::Mat();
		} else {
			throw std::out_of_range("Index out of bounds.");
		}
	}

	template<typename T>
	size_t Volume<T>::getSizeAlongDimension(Axis axis) const {
		if (axis == Axis::X) {
			return this->xSize();
		} else if (axis == Axis::Y) {
			return this->ySize();
		} else {
			return this->zSize();
		}
	}

	template<typename T>
	void Volume<T>::stop() {
		this->stopActiveProcess = true;
	}

}

#endif