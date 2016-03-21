#include <iostream>

#include <QtCore/QtCore>
#include <QtWidgets/QtWidgets>

#include "CtVolume.h"
#include "MainInterface.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define WINDOWS
#endif

#if defined WINDOWS
#include <Windows.h>	//for setting the process priority
#endif

int init(int argc, char* argv[]) {
	QApplication app(argc, argv);

	QPixmap splashImage("./sourcefiles/data/splash.png");
	QSplashScreen splash(splashImage);
	splash.show();
	app.processEvents();

	QIcon icon;
	icon.addFile("./data/icon_16.png");
	icon.addFile("./data/icon_32.png");
	icon.addFile("./data/icon_48.png");
	icon.addFile("./data/icon_64.png");
	icon.addFile("./data/icon_96.png");
	icon.addFile("./data/icon_128.png");
	icon.addFile("./data/icon_192.png");
	icon.addFile("./data/icon_256.png");
	app.setWindowIcon(icon);

	ct::MainInterface* mainInterface = new ct::MainInterface();
	mainInterface->show();
	splash.finish(mainInterface);

	//QMainWindow window;
	//window.setCentralWidget(mainInterface);
	//window.show();

	return app.exec();
}

void parseDoubleArgument(int argc, char* argv[], int index, double& output) {
	if (index < argc) {
		output = std::stod(argv[index]);
	}
}

int initConsoleMode(int argc, char* argv[]) {
	QApplication app(argc, argv);
	ct::CtVolume volume;
	bool inputProvided = false;
	bool outputProvided = false;
	bool lowerPriority = false;
	bool useCuda = volume.cudaAvailable();
	ct::FilterType filterType = ct::FilterType::RAMLAK;
	ct::IndexOrder indexOrder = ct::IndexOrder::X_FASTEST;
	QDataStream::ByteOrder byteOrder = QDataStream::LittleEndian;
	std::string filterTypeString = "Ram-Lak";
	QString input;
	QString output;
	double xmin = 0, xmax = 1, ymin = 0, ymax = 1, zmin = 0, zmax = 1;
	if (std::string(argv[1]).compare("--help") == 0 || std::string(argv[1]).compare("-h") == 0) {
		std::cout << "Usage: Cv [parameters]" << std::endl;
		std::cout << "Parameters:" << std::endl << "\t-i [path]\tFile path to the input config file. Long: --input." << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t-o [path]\tFile path for the output file. Long: --output." << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t-b \t\tOptional. Run with background priority.\n\t\t\tLong: --background." << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t-f [option] \tOptional. Sets the preprocessing filter. Options are\n\t\t\t'ramlak', 'shepplogan' and 'hann'. Long: --filter.\n\t\t\tDefault: ramlak" << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t-n \t\tOptional. Disables CUDA. Long: --nocuda." << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t-d 0,1,..,n \tOptional. Sets the cuda devices that shall be used.\n\t\t\tOption is a list of device ids seperated by comma.\n\t\t\tLong: --cudadevices.\n\t\t\tDefault: 0" << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t-d [number] \tOptional. Sets the amount of VRAM to spare in Mb. \n\t\t\tOption is a positive integer. Long: --cudasparememory.\n\t\t\tDefault: 200" << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t-e [option] \tOptional. Sets the byte order of the output. Options\n\t\t\tare 'littleendian' and 'bigendian'. Long: --byteorder.\n\t\t\tDefault: littleendian" << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t-j [option] \tOptional. Optional. Sets the index order of the output.\n\t\t\tOptions are 'zfastest' and 'xfastest'.\n\t\t\tLong: --indexorder.\n\t\t\tDefault: zfastest" << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t-h \t\tDisplay this help. Long: --help." << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t--xmin 0..1\tOptional. The lower x bound of the part of the volume\n\t\t\tthat will be reconstructed as float between 0 and 1." << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t--xmax 0..1\tOptional. The upper x bound of the part of the volume\n\t\t\tthat will be reconstructed as float between 0 and 1." << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t--ymin 0..1\tOptional. The lower y bound of the part of the volume\n\t\t\tthat will be reconstructed as float between 0 and 1." << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t--ymax 0..1\tOptional. The upper y bound of the part of the volume\n\t\t\tthat will be reconstructed as float between 0 and 1." << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t--zmin 0..1\tOptional. The lower z bound of the part of the volume\n\t\t\tthat will be reconstructed as float between 0 and 1." << std::endl;
		std::cout << "\t-----------------------------------------------------------------------" << std::endl;
		std::cout << "\t--zmax 0..1\tOptional. The upper z bound of the part of the volume\n\t\t\tthat will be reconstructed as float between 0 and 1." << std::endl;
	} else {
		for (int i = 1; i < argc; ++i) {
			if (std::string(argv[i]).compare("--xmin") == 0) {
				parseDoubleArgument(argc, argv, ++i, xmin);
			} else if (std::string(argv[i]).compare("--xmax") == 0) {
				parseDoubleArgument(argc, argv, ++i, xmax);
			} else if (std::string(argv[i]).compare("--ymin") == 0) {
				parseDoubleArgument(argc, argv, ++i, ymin);
			} else if (std::string(argv[i]).compare("--ymax") == 0) {
				parseDoubleArgument(argc, argv, ++i, ymax);
			} else if (std::string(argv[i]).compare("--zmin") == 0) {
				parseDoubleArgument(argc, argv, ++i, zmin);
			} else if (std::string(argv[i]).compare("--zmax") == 0) {
				parseDoubleArgument(argc, argv, ++i, zmax);
			} else if (std::string(argv[i]).compare("-f") == 0 || std::string(argv[i]).compare("--filter") == 0) {
				if (argc > i + 1) {
					std::string filter = argv[++i];
					std::transform(filter.begin(), filter.end(), filter.begin(), ::tolower);
					if (filter == "shepplogan") {
						filterType = ct::FilterType::SHEPP_LOGAN;
						filterTypeString = "Shepp-Logan";
					} else if (filter == "hann") {
						filterType = ct::FilterType::HANN;
						filterTypeString = "Hann";
					}
				}
			} else if (std::string(argv[i]).compare("-i") == 0 || std::string(argv[i]).compare("--input") == 0) {
				if (argc > i + 1) {
					input = QCoreApplication::arguments().at(++i);
					inputProvided = true;
				}
			} else if (std::string(argv[i]).compare("-o") == 0 || std::string(argv[i]).compare("--output") == 0) {
				if (argc > i + 1) {
					output = QCoreApplication::arguments().at(++i);
					outputProvided = true;
				}
			} else if (std::string(argv[i]).compare("-b") == 0 || std::string(argv[i]).compare("--background") == 0) {
				lowerPriority = true;
			} else if (std::string(argv[i]).compare("-n") == 0 || std::string(argv[i]).compare("--nocuda") == 0) {
				useCuda = false;
			} else if (std::string(argv[i]).compare("-d") == 0 || std::string(argv[i]).compare("--cudadevices") == 0) {
				if (++i < argc) {
					QStringList devices = QString(argv[i]).split(",");
					std::vector<int> cudaDeviceList;
					for (QString& device : devices) {
						cudaDeviceList.push_back(device.toInt());
					}
					if (cudaDeviceList.size() < 1) {
						std::cout << "When using the flag -d you have to specify at least one device ID." << std::endl;
						return 1;
					}
					volume.setActiveCudaDevices(cudaDeviceList);
				}
			} else if (std::string(argv[i]).compare("-m") == 0 || std::string(argv[i]).compare("--cudasparememory") == 0) {
				double spareMemory = 200;
				parseDoubleArgument(argc, argv, ++i, spareMemory);
				volume.setGpuSpareMemory(spareMemory);
			} else if (std::string(argv[i]).compare("-e") == 0 || std::string(argv[i]).compare("--byteorder") == 0) {
				std::string value = argv[++i];
				std::transform(value.begin(), value.end(), value.begin(), ::tolower);
				if (value == "bigendian") byteOrder = QDataStream::BigEndian;
			} else if (std::string(argv[i]).compare("-j") == 0 || std::string(argv[i]).compare("--indexOrder") == 0) {
				std::string value = argv[++i];
				std::transform(value.begin(), value.end(), value.begin(), ::tolower);
				if (value == "zfastest") indexOrder = ct::IndexOrder::Z_FASTEST;
			} else {
				std::cout << "Unknown or misplaced parameter " << argv[i] << "." << std::endl;
				return 1;
			}
		}
		if (inputProvided && outputProvided) {
#if defined WINDOWS
			if (lowerPriority) {
				//lower priority of process
				if (!SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_BEGIN)) {
					std::cout << "Could not set the priority.";
				}
			}
#endif
			std::vector<std::string> cudaDeviceNames = volume.getCudaDeviceList();
			QVector<int> activeCudaDevices = QVector<int>::fromStdVector(volume.getActiveCudaDevices());
			std::cout << std::endl << "Beginning reconstruction." << std::endl;
			std::cout << "\tInput:\t\t\t" << input.toStdString() << std::endl;
			std::cout << "\tOutput:\t\t\t" << output.toStdString() << std::endl;
			std::cout << "\tFilter type:\t\t" << filterTypeString << std::endl;
			std::cout << "\tUsing CUDA:\t\t" << (useCuda ? "YES" : "NO") << std::endl;
			if (volume.cudaAvailable()) {
				std::cout << "\tCUDA devices:" << std::endl;
				for (int i = 0; i < cudaDeviceNames.size(); ++i) {
					std::cout << "\t\t" << "[" << (activeCudaDevices.indexOf(i) >= 0 && useCuda ? "X" : " ") << "] " << cudaDeviceNames[i] << std::endl;
				}
			}
			std::cout << "\tVolume bounds:";
			std::cout << "\t\tx: [" << std::to_string(xmin) << " .. " << std::to_string(xmax) << "]" << std::endl;;
			std::cout << "\t\t\t\ty: [" << std::to_string(ymin) << " .. " << std::to_string(ymax) << "]" << std::endl;
			std::cout << "\t\t\t\tz: [" << std::to_string(zmin) << " .. " << std::to_string(zmax) << "]" << std::endl;
			std::cout << "\tByte order:\t\t" << (byteOrder == QDataStream::LittleEndian ? "Little endian" : "Big endian") << std::endl;
			std::cout << "\tByte order:\t\t" << (indexOrder == ct::IndexOrder::Z_FASTEST ? "Z fastest" : "X fastest") << std::endl;
			std::cout << std::endl;
			volume.sinogramFromImages(input);
			volume.setVolumeBounds(xmin, xmax, ymin, ymax, zmin, zmax);
			std::cout << std::endl << "The resulting volume dimensions will be:" << std::endl << std::endl << "\t" << volume.getXSize() << "x" << volume.getYSize() << "x" << volume.getZSize() << " (x:y:z)" << std::endl << std::endl;
			volume.setFrequencyFilterType(filterType);
			volume.reconstructVolume();
			volume.saveVolumeToBinaryFile(output, indexOrder, byteOrder);
		} else {
			std::cout << "You must provide a file path to the config file as input as well as a file path for the output." << std::endl;
			return 1;
		}
	}
	return 0;
}

int main(int argc, char* argv[]) {
	if (argc >= 2) {
		return initConsoleMode(argc, argv);
	} else {
		std::cout << "Launching in GUI mode." << std::endl;
		return init(argc, argv);
	}
	return 0;
}