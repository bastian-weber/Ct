
#include <iostream>
#include "CtVolume.h"

int main(){
	CtVolume myVolume("sourcefiles/data/skullPhantom");
	myVolume.displaySinogram();	
	myVolume.reconstructVolume();
	myVolume.saveVolumeToBinaryFile("volume.raw");
	system("pause");
	return 0;
}