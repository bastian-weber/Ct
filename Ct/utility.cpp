#include "utility.h"

namespace utility {

	std::shared_ptr<std::vector<char>> readFileIntoBuffer(QString const& path) {
		////some Qt code that also does the trick
		//QFile file(path);
		//std::vector<char> buffer;
		//buffer.resize(file.size());
		//if (!file.open(QIODevice::ReadOnly)) {
		//	return std::vector<char>();
		//}
		//file.read(buffer.data(), file.size());
		//file.close();

#ifdef Q_OS_WIN
		//wchar for utf-16
		std::ifstream file(path.toStdWString(), std::iostream::binary);
#else
		//char for utf-8
		std::ifstream file(path.toStdString(), std::iostream::binary);
#endif
		if (!file.good()) {
			return std::shared_ptr<std::vector<char>>(new std::vector<char>());
		}
		file.exceptions(std::ifstream::badbit | std::ifstream::failbit | std::ifstream::eofbit);
		file.seekg(0, std::ios::end);
		std::streampos length(file.tellg());
		std::shared_ptr<std::vector<char>> buffer(new std::vector<char>(static_cast<std::size_t>(length)));
		if (static_cast<std::size_t>(length) == 0) {
			return std::shared_ptr<std::vector<char>>(new std::vector<char>());
		}
		file.seekg(0, std::ios::beg);
		try {
			file.read(buffer->data(), static_cast<std::size_t>(length));
		} catch (...) {
			return std::shared_ptr<std::vector<char>>(new std::vector<char>());
		}
		file.close();
		return buffer;
	}

	bool isCharCompatible(QString const& string) {
#ifdef Q_OS_WIN
		bool isCharCompatible = true;
		for (QString::ConstIterator i = string.begin(); i != string.end(); ++i) {
			isCharCompatible = isCharCompatible && (i->unicode() < 128);
		}
		return isCharCompatible;
#else
		return true;
#endif
	}
}