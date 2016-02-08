#include <iostream>

__global__ void mykernel(void) {
}

int test(void) {
	mykernel<<<1,1>>>();
	std::cout<<"Hello World!"<<std::endl;
	return 0;
}
