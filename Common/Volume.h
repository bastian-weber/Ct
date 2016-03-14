//note: this class is header-only

#ifndef CT_VOLUME
#define CT_VOLUME

#include <atomic>

//OpenCV
#include <opencv2/core/core.hpp>				//core functionality of OpenCV

//Qt
#include <QtCore/QtCore>

#include "CompletionStatus.h"

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

	//base class for signals (signals do not work in template class)
	class VolumeSignalsSlots : public QObject {
		Q_OBJECT
	public:
		virtual ~VolumeSignalsSlots() = default;
	signals:
		void savingProgress(double percentage) const;
		void savingFinished(CompletionStatus status = CompletionStatus::success()) const;
		void loadingProgress(double percentage) const;
		void loadingFinished(CompletionStatus status = CompletionStatus::success()) const;
	};

	template <typename T>
	class Volume : public std::vector<std::vector<std::vector<T>>>, public VolumeSignalsSlots {
	public:
		Volume() = default;
		Volume(size_t xSize, size_t ySize, size_t zSize, T defaultValue = 0);
		Volume& operator=(Volume const& other) = delete;
		void reinitialise(size_t xSize,											//resizes the volume to the given dimensions and sets all elements to the given value
						  size_t ySize, 
						  size_t zSize, 
						  T defaultValue = 0);
		template <typename U>
		bool loadFromBinaryFile(QString const& filename,						//reads a volume from a binary file
								size_t xSize,
								size_t ySize,
								size_t zSize,
								QDataStream::FloatingPointPrecision floatingPointPrecision = QDataStream::SinglePrecision,
								QDataStream::ByteOrder byteOrder = QDataStream::LittleEndian,
								T* minValue = nullptr,
								T* maxValue = nullptr);
		bool saveToBinaryFile(QString const& filename,						//saves the volume to a binary file with the given filename
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
		T& at(size_t x, size_t y, size_t z);
		T const& at(size_t x, size_t y, size_t z) const;
		//setters
		void setEmitSignals(bool value);
	private:
		using std::vector<std::vector<std::vector<T>>>::operator[];
		using std::vector<std::vector<std::vector<T>>>::size;

		bool emitSignals = true;												//if true the object emits qt signals in certain functions
		mutable std::atomic<bool> stopActiveProcess{ false };
	};

	//=========================================== IMPLEMENTATION ===========================================\\

	template <typename T>
	Volume<T>::Volume(size_t xSize, size_t ySize, size_t zSize, T defaultValue) 
	: std::vector<std::vector<std::vector<T>>>(xSize, std::vector<std::vector<T>>(ySize, std::vector<T>(zSize, defaultValue))) { }

	template <typename T>
	void Volume<T>::reinitialise(size_t xSize, size_t ySize, size_t zSize, T defaultValue) {
		this->clear();
		this->resize(xSize, std::vector<std::vector<T>>(ySize, std::vector<T>(zSize, defaultValue)));
	}

	template<typename T>
	void Volume<T>::setEmitSignals(bool value) {
		this->emitSignals = value;
	}

	template<typename T>
	bool Volume<T>::getEmitSignals() const {
		return this->emitSignals;
	}

	template<typename T>
	size_t Volume<T>::xSize() const {
		return this->size();
	}

	template<typename T>
	size_t Volume<T>::ySize() const {
		if (this->size() != 0) {
			return (*this)[0].size();
		}
		return 0;
	}

	template<typename T>
	size_t Volume<T>::zSize() const {
		if (this->size() != 0 && (*this)[0].size()) {
			return (*this)[0][0].size();
		}
		return 0;
	}

	template<typename T>
	inline T& Volume<T>::at(size_t x, size_t y, size_t z) {
		return (*this)[x][y][z];
	}

	template<typename T>
	inline T const& Volume<T>::at(size_t x, size_t y, size_t z) const {
		return (*this)[x][y][z];
	}

	template <typename T>
	template <typename U>
	bool Volume<T>::loadFromBinaryFile(QString const& filename, size_t xSize, size_t ySize, size_t zSize, QDataStream::FloatingPointPrecision floatingPointPrecision, QDataStream::ByteOrder byteOrder, T* minValue, T* maxValue) {
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
		T min = std::numeric_limits<T>::max();
		T max = std::numeric_limits<T>::lowest();
		U tmp;
		T converted;
		for (int x = 0; x < this->xSize(); ++x) {
			if (this->stopActiveProcess) {
				this->clear();
				std::cout << "User interrupted. Stopping." << std::endl;
				if (this->emitSignals) emit(loadingFinished(CompletionStatus::interrupted()));
				return false;
			}
			double percentage = floor(double(x) / double(this->xSize()) * 100 + 0.5);
			if (this->emitSignals) emit(loadingProgress(percentage));
			for (int y = 0; y < this->ySize(); ++y) {
				for (int z = 0; z < this->zSize(); ++z) {
					//load one U of data
					in >> tmp;
					converted = static_cast<T>(tmp);
					if (converted < min) min = converted;
					if (converted > max) max = converted;
					this->at(x, y, z) = converted;
				}
			}
		}
		file.close();
		if (minValue != nullptr) *minValue = min;
		if (maxValue != nullptr) *maxValue = max;
		if (this->emitSignals) emit(loadingFinished());
		return true;
	}

	template <typename T>
	bool Volume<T>::saveToBinaryFile(QString const& filename, QDataStream::FloatingPointPrecision floatingPointPrecision, QDataStream::ByteOrder byteOrder) const {
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
				for (int x = 0; x < this->xSize(); ++x) {
					if (this->stopActiveProcess) {
						std::cout << "User interrupted. Stopping." << std::endl;
						if (this->emitSignals) emit(savingFinished(CompletionStatus::interrupted()));
						return false;
					}
					double percentage = floor(double(x) / double(this->xSize()) * 100 + 0.5);
					if (this->emitSignals) emit(savingProgress(percentage));
					for (int y = 0; y < this->ySize(); ++y) {
						for (int z = 0; z < this->zSize(); ++z) {
							//save one T of data
							out << this->at(x, y, z);
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
		if (this->size() == 0) return cv::Mat();
		if (index >= 0 && ((axis == Axis::X && index < this->xSize()) || (axis == Axis::Y && index < this->zSize()) || (axis == Axis::Z && index < this->zSize()))) {
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
				std::function<void(int, int, float*)> setPixel;
				
				switch (axis) {
					case Axis::X:
						if (type == CoordinateSystemOrientation::LEFT_HANDED) {
							setPixel = [&](int row, int column, float* ptr) {
								ptr[column] = static_cast<float>(this->at(index, column, result.rows - 1 - row));
							};
						} else {
							setPixel = [&](int row, int column, float* ptr) {
								ptr[column] = static_cast<float>(this->at(index, result.cols - 1 - column, result.rows - 1 - row));
							};
						}
						break;
					case Axis::Y:
						if (type == CoordinateSystemOrientation::LEFT_HANDED) {
							setPixel = [&](int row, int column, float* ptr) {
								ptr[column] = static_cast<float>(this->at(result.cols - 1 - column, index, result.rows - 1 - row));
							};
						} else {
							setPixel = [&](int row, int column, float* ptr) {
								ptr[column] = static_cast<float>(this->at(column, index, result.rows - 1 - row));
							};
						}
						break;
					case Axis::Z:
						if (type == CoordinateSystemOrientation::LEFT_HANDED) {
							setPixel = [&](int row, int column, float* ptr) {
								ptr[column] = static_cast<float>(this->at(row, column, index));
							};
						} else {
							setPixel = [&](int row, int column, float* ptr) {
								ptr[column] = static_cast<float>(this->at(row, result.cols - 1 - column, index));
							};
						}
						break;
				}

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
		if (this->size() != 0) {
			if (axis == Axis::X) {
				return this->xSize();
			} else if (axis == Axis::Y) {
				return this->ySize();
			} else {
				return this->zSize();
			}
		}
		return 0;
	}

	template<typename T>
	void Volume<T>::stop() {
		this->stopActiveProcess = true;
	}

}

#endif