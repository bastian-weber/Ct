//note: this class is header-only

#ifndef CT_VOLUME
#define CT_VOLUME

#include <atomic>

//Qt
#include <QtCore/QtCore>

#include "CompletionStatus.h"

namespace ct {

	//base class for signals (signals do not work in template class)
	class VolumeSignalsSlots : public QObject {
		Q_OBJECT
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
		void reinitialise(size_t xSize, size_t ySize, size_t zSize, T defaultValue = 0);
		bool saveToBinaryFile(std::string const& filename) const;
		void stop();
		void setEmitSignals(bool value);
		bool getEmitSignals() const;
	private:
		bool emitSignals = true;											//if true the object emits qt signals in certain functions
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
	inline bool Volume<T>::getEmitSignals() const {
		return this->emitSignals;
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
				if (std::is_same<T, double>::value || std::is_same<T, long double>::value) {
					out.setFloatingPointPrecision(QDataStream::DoublePrecision);
				} else {
					out.setFloatingPointPrecision(QDataStream::SinglePrecision);
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
	inline void Volume<T>::stop() {
		this->stopActiveProcess = true;
	}

}



#endif