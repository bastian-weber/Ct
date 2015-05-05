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

	ct::MainInterface* mainInterface = new ct::MainInterface();
	mainInterface->show();

	return app.exec();
}

int main(int argc, char* argv[]) {

	bool showSinogram = false;
	bool inputProvided = false;
	bool outputProvided = false;
	bool lowerPriority = false;
	std::string input;
	std::string output;

	if (argc >= 2) {
		if (std::string(argv[1]).compare("--help") == 0 || std::string(argv[1]).compare("-h") == 0) {
			std::cout << "Usage: Cv [parameters]" << std::endl;
			std::cout << "Parameters:" << std::endl << "\t-i [Path]\tFile path to the input config file. Long: --input." << std::endl;
			std::cout << "\t-o [Path]\tFile path for the output file. Long: --output." << std::endl;
			std::cout << "\t-d \t\tOptional. Display the sinogram. Long: --display." << std::endl;
			std::cout << "\t-s \t\tOptional. Run only singlethreaded. \n\t\t\tLong: --singlethreaded." << std::endl;
			std::cout << "\t-h \t\tDisplay this help. Long: --help." << std::endl;
		} else {
			for (int i = 1; i < argc; ++i) {
				if (std::string(argv[i]).compare("-d") == 0 || std::string(argv[i]).compare("--display") == 0) {
					showSinogram = true;
				} else if (std::string(argv[i]).compare("-i") == 0 || std::string(argv[i]).compare("--input") == 0) {
					input = argv[++i];
					inputProvided = true;
				} else if (std::string(argv[i]).compare("-o") == 0 || std::string(argv[i]).compare("--output") == 0) {
					output = argv[++i];
					outputProvided = true;
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
				std::cout << "\tDisplay sinogram:\t" << ((showSinogram) ? "YES" : "NO") << std::endl << std::endl;;
				ct::CtVolume myVolume(input, ct::CtVolume::RAMLAK);
				if (showSinogram)myVolume.displaySinogram(true);
				myVolume.reconstructVolume();
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