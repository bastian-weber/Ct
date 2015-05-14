This project implements the Feldkamp algorithm for reconstructing a computer tomography density volume from x-ray images.

With cmake you can automatically create a Visual Studio project on Windows or a makefile on Linux.

============================================ Libraries ============================================

The required libraries are:

	1. OpenCV (currently 2.4.10)
			http://opencv.org/downloads.html
	2. FFTW 3 (float precision variant)
			http://www.fftw.org/download.html
	3. Qt (currently 5.4.1)

============================================= Windows =============================================

	-	For FFTW import libraries (.lib) have to be created using the Visual Studio Native Prompt. See the fftw readme for that.
	-	Qt can be downloaded precompiled as installer. QtCreator is _not_ required.
	-	OpenCV can also be downloaded precompiled

In cmake just set the necessary paths to the libraries and click configure and generate. The dlls should automatically be copied
to the Release and Debug directory.

============================================== Linux ==============================================

	-	OpenCV can be compiled using cmake and make
	-	FFTW can be built by calling configure (make sure to set the float precision flag) and then make
	-	Qt can be downloaded precompiled as an installer version

Then just use cmake as on Windows.