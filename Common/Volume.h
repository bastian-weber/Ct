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

	//base class for signals (signals do not work in template class)
	class VolumeSignalsSlots : public QObject {
		Q_OBJECT
	public:
		virtual ~VolumeSignalsSlots() = default;
	signals:
		void savingProgress(double percentage) const;
		void savingFinished(CompletionStatus status = CompletionStatus::success()) const;
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
		bool saveToBinaryFile(std::string const& filename) const;				//saves the volume to a binary file with the given filename
		cv::Mat getVolumeCrossSection(Axis axis, size_t index) const;			//returns a cross section through the volume as image
		size_t getSizeAlongDimension(Axis axis) const;							//returns the size along the axis axis
		void stop();															//stops the saving function
		//getters
		bool getEmitSignals() const;
		size_t xSize() const;
		size_t ySize() const;
		size_t zSize() const;
		//setters
		void setEmitSignals(bool value);
	private:
		bool emitSignals = true;												//if true the object emits qt signals in certain functions
		mutable std::atomic<bool> stopActiveProcess{ false };
	};

	//=========================================== IMPLEMENTATION ===========================================\\

	template <typename T>
	Volume<T>::Volume(size_t xSize, size_t ySize, size_t zSize, T defaultValue = 0) 
	: std::vector<std::vector<std::vector<T>>>(xSize, std::vector<std::vector<T>>(ySize, std::vector<T>(zSize, defaultValue))) { }

	template <typename T>
	void Volume<T>::reinitialise(size_t xSize, size_t ySize, size_t zSize, T defaultValue = 0) {
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

	template <typename T>
	bool Volume<T>::saveToBinaryFile(std::string const& filename) const {
		this->stopActiveProcess = false;
		if (this->size() > 0 && (*this)[0].size() > 0 && (*this)[0][0].size() > 0) {
			{
				//write binary file
				QFile file(filename.c_str());
				if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
					std::cout << "Could not open the file. Maybe your path does not exist. No files were written." << std::endl;
					if(this->emitSignals) emit(savingFinished(CompletionStatus::error("Could not open the file. Maybe your path does not exist. No files were written.")));
					return false;
				}
				QDataStream out(&file);
				if (std::is_floating_point<T>::value) {
					if (sizeof(T) > 4) {
						out.setFloatingPointPrecision(QDataStream::DoublePrecision);
					} else {
						out.setFloatingPointPrecision(QDataStream::SinglePrecision);
					}
				}
				out.setByteOrder(QDataStream::LittleEndian);
				//iterate through the volume
				for (int x = 0; x < this->size(); ++x) {
					if (this->stopActiveProcess) break;
					double percentage = floor(double(x) / double(this->size()) * 100 + 0.5);
					if (this->emitSignals) emit(savingProgress(percentage));
					for (int y = 0; y < (*this)[0].size(); ++y) {
						for (int z = 0; z < (*this)[0][0].size(); ++z) {
							//save one T of data
							out << (*this)[x][y][z];
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
		if (this->stopActiveProcess) {
			std::cout << "User interrupted. Stopping." << std::endl;
			if (this->emitSignals) emit(savingFinished(CompletionStatus::interrupted()));
			return false;
		} else {
			std::cout << "Volume successfully saved." << std::endl;
			if (this->emitSignals) emit(savingFinished());
		}
		return true;
	}

	template<typename T>
	cv::Mat Volume<T>::getVolumeCrossSection(Axis axis, size_t index) const {
		if (this->size() == 0) return cv::Mat();
		size_t xMax = this->size();
		size_t yMax = (*this)[0].size();
		size_t zMax = (*this)[0][0].size();
		if (index >= 0 && ((axis == Axis::X && index < xMax) || (axis == Axis::Y && index < yMax) || (axis == Axis::Z && index < zMax))) {
			if (this->size() > 0 && (*this)[0].size() > 0 && (*this)[0][0].size() > 0) {
				size_t uSize;
				size_t vSize;
				if (axis == Axis::X) {
					uSize = yMax;
					vSize = zMax;
				} else if (axis == Axis::Y) {
					uSize = xMax;
					vSize = zMax;
				} else {
					uSize = xMax;
					vSize = yMax;
				}

				cv::Mat result(vSize, uSize, CV_32FC1);
				float* ptr;
				if (axis == Axis::X) {
#pragma omp parallel for private(ptr)
					for (int row = 0; row < result.rows; ++row) {
						ptr = result.ptr<float>(row);
						for (int column = 0; column < result.cols; ++column) {
							ptr[column] = static_cast<float>((*this)[index][column][result.rows - 1 - row]);
						}
					}
				} else if (axis == Axis::Y) {
#pragma omp parallel for private(ptr)
					for (int row = 0; row < result.rows; ++row) {
						ptr = result.ptr<float>(row);
						for (int column = 0; column < result.cols; ++column) {
							ptr[column] = static_cast<float>((*this)[column][index][result.rows - 1 - row]);
						}
					}
				} else {
#pragma omp parallel for private(ptr)
					for (int row = 0; row < result.rows; ++row) {
						ptr = result.ptr<float>(row);
						for (int column = 0; column < result.cols; ++column) {
							ptr[column] = static_cast<float>((*this)[column][row][index]);
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
	}

	template<typename T>
	void Volume<T>::stop() {
		this->stopActiveProcess = true;
	}

}

#endif