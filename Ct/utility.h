#ifndef UTILITY
#define UTILITY

#include <vector>
#include <fstream>
#include <memory>

//Qt
#include <QtCore>

namespace utility {

	std::shared_ptr<std::vector<char>> readFileIntoBuffer(QString const& path);

	bool isCharCompatible(QString const& string);

}

#endif