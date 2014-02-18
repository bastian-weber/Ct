#include <iostream>
#include "CtVolume.h"

int main(){
	CtVolume myVolume("sourcefiles/data/skullPhantom", "sourcefiles/data/skullPhantom/angles.csv", CtVolume::TIF, CtVolume::RAMLAK);
	myVolume.displaySinogram(true);	
	myVolume.reconstructVolume(CtVolume::MULTITHREADED);
	myVolume.saveVolumeToBinaryFile("G:/Desktop/volume.raw");
	return 0;
}