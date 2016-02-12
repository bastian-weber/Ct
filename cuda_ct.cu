#include "cuda_ct.h"

namespace ct {

	namespace cuda {

		__global__ void mykernel(void) { }

		int test(void) {
			mykernel <<<1, 1 >>>();
			return 0;
		}

	}

}