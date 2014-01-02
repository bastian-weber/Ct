#include <iostream>
#include "CtVolume.h"

int main(){
	CtVolume myVolume("sourcefiles/data/skullPhantom");
	myVolume.displaySinogram();

	return 0;
}