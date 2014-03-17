#include <iostream>
#include <Windows.h>
#include "CtVolume.h"

int main(int argc, char* argv[]){
	//if (!SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_BEGIN)){
	//	std::cout << "could not set the priority";
	//}

	CtVolume::ThreadingType tt = CtVolume::MULTITHREADED;
	bool showSinogram = false;
	bool inputProvided = false;
	bool outputProvided = false;
	std::string input;
	std::string output;

	if (argc >= 2){
		if (std::string(argv[1]).compare("--help") == 0 || std::string(argv[1]).compare("-h") == 0){
			std::cout << "Usage: Cv [parameters]" << std::endl;
			std::cout << "Parameters:" << std::endl << "\t-i [Path]\tFile path to the input config file. Long: --input." << std::endl;
			std::cout<<"\t-o [Path]\tFile path for the output file. Long: --output." << std::endl;
			std::cout<<"\t-d \t\tOptional. Display the sinogram. Long: --display." << std::endl;
			std::cout<<"\t-s \t\tOptional. Run only singlethreaded. \n\t\t\tLong: --singlethreaded." << std::endl;
			std::cout<<"\t-h \t\tDisplay this help. Long: --help." << std::endl;
		} else{
			for (int i = 1; i < argc; ++i){
				if (std::string(argv[i]).compare("-s") == 0 || std::string(argv[i]).compare("--singlethreaded") == 0){
					tt = CtVolume::SINGLETHREADED;
				} else if (std::string(argv[i]).compare("-d") == 0 || std::string(argv[i]).compare("--display") == 0){
					showSinogram = true;
				} else if (std::string(argv[i]).compare("-i") == 0 || std::string(argv[i]).compare("--input") == 0){
					input = argv[++i];
					inputProvided = true;
				} else if (std::string(argv[i]).compare("-o") == 0 || std::string(argv[i]).compare("--output") == 0){
					output = argv[++i];
					outputProvided = true;
				} else{
					std::cout << "Unknown or misplaced parameter " << argv[i] << "." << std::endl;
					return 1;
				}
			}
			if (inputProvided && outputProvided){
				std::cout << std::endl << "Beginning reconstruction." << std::endl;
				std::cout << "\tInput:\t\t\t" << input<<std::endl;
				std::cout << "\tOutput:\t\t\t" << output<<std::endl;
				std::cout << "\tMultithreaded:\t\t" << ((tt == CtVolume::MULTITHREADED) ? "YES" : "NO") << std::endl;
				std::cout << "\tDisplay sinogram:\t" << ((showSinogram) ? "YES" : "NO") << std::endl << std::endl;;
				CtVolume myVolume(input, CtVolume::RAMLAK);
				if(showSinogram)myVolume.displaySinogram(true);	
				myVolume.reconstructVolume(tt);
				myVolume.saveVolumeToBinaryFile(output);
			} else{
				std::cout << "You must provide a file path to the config file as input as well as a file path for the output." << std::endl;
				return 1;
			}
		}
	} else{
		std::cout << "Invalid argument count. Use --help for help." << std::endl;
		return 1;
	}
	//CtVolume myVolume("G:/Desktop/Turnschuh_1200/angles.csv", CtVolume::RAMLAK);
	//myVolume.displaySinogram(true);	
	//myVolume.reconstructVolume(CtVolume::MULTITHREADED);
	//myVolume.saveVolumeToBinaryFile("G:/Desktop/volume.raw");
	return 0;
}