#include <iostream>
#include "CtVolume.h"

int main(){
	CtVolume myVolume("G:/Desktop/Turnschuh2", "G:/Desktop/Turnschuh2/angles2.csv", CtVolume::TIF, CtVolume::RAMLAK);
	myVolume.displaySinogram(true);	
	myVolume.reconstructVolume(CtVolume::SINGLETHREADED);
	myVolume.saveVolumeToBinaryFile("G:/Desktop/volume.raw");
	return 0;
}