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

int main(int argc, char* argv[]) {
	bool inputProvided = false;
	bool outputProvided = false;
	bool lowerPriority = false;
	ct::FilterType filterType = ct::FilterType::RAMLAK;
	std::string filterTypeString = "Ram-Lak";
	std::string input;
	std::string output;
	double xmin = 0, xmax = 1, ymin = 0, ymax = 1, zmin = 0, zmax = 1;

	if (argc >= 2) {
		if (std::string(argv[1]).compare("--help") == 0 || std::string(argv[1]).compare("-h") == 0) {
			std::cout << "Usage: Cv [parameters]" << std::endl;
			std::cout << "Parameters:" << std::endl << "\t-i [path]\tFile path to the input config file. Long: --input." << std::endl;
			std::cout << "\t-----------------------------------------------------------------------" << std::endl;
			std::cout << "\t-o [path]\tFile path for the output file. Long: --output." << std::endl;
			std::cout << "\t-----------------------------------------------------------------------" << std::endl;
			std::cout << "\t-b \t\tOptional. Run with background priority.\n\t\t\tLong: --background." << std::endl;
			std::cout << "\t-----------------------------------------------------------------------" << std::endl;
			std::cout << "\t-f [option] \tOptional. Sets the preprocessing filter. Options are\n\t\t\t'ramlak', 'shepplogan' and 'hann'. Long: --filter." << std::endl;
			std::cout << "\t-----------------------------------------------------------------------" << std::endl;
			std::cout << "\t-h \t\tDisplay this help. Long: --help." << std::endl;
			std::cout << "\t-----------------------------------------------------------------------" << std::endl;
			std::cout << "\t--xmin [0..1]\tOptional. The lower x bound of the part of the volume\n\t\t\tthat will be reconstructed as float between 0 and 1." << std::endl;
			std::cout << "\t-----------------------------------------------------------------------" << std::endl;
			std::cout << "\t--xmax [0..1]\tOptional. The upper x bound of the part of the volume\n\t\t\tthat will be reconstructed as float between 0 and 1." << std::endl;
			std::cout << "\t-----------------------------------------------------------------------" << std::endl;
			std::cout << "\t--ymin [0..1]\tOptional. The lower y bound of the part of the volume\n\t\t\tthat will be reconstructed as float between 0 and 1." << std::endl;
			std::cout << "\t-----------------------------------------------------------------------" << std::endl;
			std::cout << "\t--ymax [0..1]\tOptional. The upper y bound of the part of the volume\n\t\t\tthat will be reconstructed as float between 0 and 1." << std::endl;
			std::cout << "\t-----------------------------------------------------------------------" << std::endl;
			std::cout << "\t--zmin [0..1]\tOptional. The lower z bound of the part of the volume\n\t\t\tthat will be reconstructed as float between 0 and 1." << std::endl;
			std::cout << "\t-----------------------------------------------------------------------" << std::endl;
			std::cout << "\t--zmax [0..1]\tOptional. The upper z bound of the part of the volume\n\t\t\tthat will be reconstructed as float between 0 and 1." << std::endl;
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
						input = argv[++i];
						inputProvided = true;
					}
				} else if (std::string(argv[i]).compare("-o") == 0 || std::string(argv[i]).compare("--output") == 0) {
					if (argc > i + 1) {
						output = argv[++i];
						outputProvided = true;
					}
				} else if (std::string(argv[i]).compare("-b") == 0 || std::string(argv[i]).compare("--background") == 0) {
					lowerPriority = true;
				} else {
					std::cout << "Unknown or misplaced parameter " << argv[i] << "." << std::endl;
					return 1;
				}
			}
			if (inputProvided && outputProvided) {
#if defined WINDOWS
				if (lowerPriority) {
					//lower priority of process
					if (!SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_BEGIN)){
						std::cout << "Could not set the priority.";
					}
				}
#endif
				std::cout << std::endl << "Beginning reconstruction." << std::endl;
				std::cout << "\tInput:\t\t\t" << input << std::endl;
				std::cout << "\tOutput:\t\t\t" << output << std::endl;
				std::cout << "\tFilter type:\t\t" << filterTypeString << std::endl;
				std::cout << "\tVolume bounds:";
				std::cout << "\t\tx: [" << std::to_string(xmin) << " .. " << std::to_string(xmax) << "]" << std::endl;;
				std::cout << "\t\t\t\ty: [" << std::to_string(ymin) << " .. " << std::to_string(ymax) << "]" << std::endl;
				std::cout << "\t\t\t\tz: [" << std::to_string(zmin) << " .. " << std::to_string(zmax) << "]" << std::endl;
				std::cout << std::endl;
				ct::CtVolume myVolume(input);
				myVolume.setVolumeBounds(xmin, xmax, ymin, ymax, zmin, zmax);
				std::cout << std::endl << "The resulting volume dimensions will be:" << std::endl << std::endl << "\t" << myVolume.getXSize() << "x" << myVolume.getYSize() << "x" << myVolume.getZSize() << " (x:y:z)" << std::endl << std::endl;
				myVolume.reconstructVolume(filterType);
				myVolume.saveVolumeToBinaryFile(output);
			} else {
				std::cout << "You must provide a file path to the config file as input as well as a file path for the output." << std::endl;
				return 1;
			}
		}
	} else {
		std::cout << "Launching in GUI mode." << std::endl;
		return init(argc, argv);
	}
	//CtVolume myVolume("G:/Desktop/Turnschuh_1200/angles.csv", CtVolume::RAMLAK);
	//myVolume.displaySinogram(true);	
	//myVolume.reconstructVolume(CtVolume::MULTITHREADED);
	//myVolume.saveVolumeToBinaryFile("G:/Desktop/volume.raw");
	return 0;
}