#include <iostream>
#include "CtVolume.h"

////converts an image to 32 bit float
////only unsigned types are allowed as input
//void convertTo32bit(cv::Mat& img){
//	CV_Assert(img.depth() == CV_8U || img.depth() == CV_16U || img.depth() == CV_32F);
//	if (img.depth() == CV_8U){
//		img.convertTo(img, CV_32F, 1.0 / (float)pow(2, 8));
//	} else if (img.depth() == CV_16U){
//		img.convertTo(img, CV_32F, 1.0 / (float)pow(2, 16));
//	}
//}
//
//int fftCoordToIndex(int coord, int size){
//	if (coord < 0)return size + coord;
//	return coord;
//}
//
//void lowpassFilter(cv::Mat& image){
//	CV_Assert(image.depth() == CV_32F);
//
//	//FFT
//	
//	const int R = image.rows;
//	const int C = image.cols;
//
//	std::vector<std::vector<std::complex<double>>> result(R, std::vector<std::complex<double>>(C));
//	double *in;
//	in = (double*)fftw_malloc(sizeof(double)* R * C);
//	int k = 0;
//	float* ptr;
//	for (int row = 0; row < R; ++row){
//		ptr = image.ptr<float>(row);
//		for (int column = 0; column < C; ++column){
//			in[k] = ptr[column];
//			++k;
//		}
//	}
//	fftw_complex *out;
//	int nyquist = (C / 2) + 1;
//	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)* R * C);
//	fftw_plan p;
//	p = fftw_plan_dft_r2c_2d(R, C, in, out, FFTW_ESTIMATE);
//	fftw_execute(p);
//	fftw_destroy_plan(p);
//
//	k = 0;
//	for (int row = 0; row < R; ++row){
//		for (int column = 0; column < nyquist; ++column){
//			out[k][0] = out[k][0] / (R*C);
//			out[k][1] = out[k][1] / (R*C);
//			++k;
//		}
//	}
//
//	//removing the low frequencies
//
//	double cutoffRatio = 0.05;
//	int uCutoff = (cutoffRatio / 2.0)*C;
//	int vCutoff = (cutoffRatio / 2.0)*R;
//	for (int row = -vCutoff; row <= vCutoff; ++row){
//		for (int column = 0; column <= uCutoff; ++column){
//			out[fftCoordToIndex(row, R)*nyquist + column][0] = 0;
//			out[fftCoordToIndex(row, R)*nyquist + column][1] = 0;
//		}
//	}
//
//	//reverse FFT
//
//	p = fftw_plan_dft_c2r_2d(R, C, out, in, FFTW_ESTIMATE);
//	fftw_execute(p);
//	fftw_destroy_plan(p);
//	k = 0;
//	for (int row = 0; row < R; ++row){
//		ptr = image.ptr<float>(row);
//		for (int column = 0; column < C; ++column){
//			ptr[column] = in[k];
//			++k;
//		}
//	}
//
//	fftw_free(out);
//	fftw_free(in);
//}

int main(){
	CtVolume myVolume("sourcefiles/data/skullPhantom", "sourcefiles/data/skullPhantom/angles.csv");
	//myVolume.displaySinogram();	
	myVolume.reconstructVolume(CtVolume::MULTITHREADED);
	myVolume.saveVolumeToBinaryFile("G:/Desktop/volume.raw");
	system("pause");
	return 0;
}