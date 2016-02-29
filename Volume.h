#ifndef CT_VOLUME
#define CT_VOLUME

//Qt
#include <QtCore/QtCore>

namespace ct {

	template <typename T>
	class Volume : public std::vector<std::vector<std::vector<T>>> {
	public:
		Volume(size_t xSize, size_t ySize, size_t zSize, T defaultValue = 0);
	private:
	};

	//implementation of template functions

	template <typename T>
	Volume<T>::Volume(size_t xSize, size_t ySize, size_t zSize, T defaultValue = 0) 
	: std::vector<std::vector<std::vector<T>>>(xSize, std::vector<std::vector<T>>(ySize, std::vector<T>(zSize, defaultValue))) {
	}

}



#endif