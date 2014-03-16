#include <iostream>
#include <Windows.h>
#include "CtVolume.h"

int main(){
	//if (!SetPriorityClass(GetCurrentProcess(), PROCESS_MODE_BACKGROUND_BEGIN)){
	//	std::cout << "could not set the priority";
	//}
	CtVolume myVolume("G:/Desktop/Turnschuh_256/angles.csv", CtVolume::RAMLAK);
	//myVolume.displaySinogram(true);	
	myVolume.reconstructVolume(CtVolume::MULTITHREADED);
	myVolume.saveVolumeToBinaryFile("G:/Desktop/volume.raw");
	return 0;
}