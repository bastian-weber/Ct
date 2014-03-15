#include <iostream>
#include "CtVolume.h"

int main(){
	CtVolume myVolume("G:/Desktop/Turnschuh2/angles.csv", CtVolume::RAMLAK);
	myVolume.displaySinogram(true);	
	myVolume.reconstructVolume(CtVolume::MULTITHREADED);
	myVolume.saveVolumeToBinaryFile("G:/Desktop/volume.raw");
	return 0;
}