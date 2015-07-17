This project implements the Feldkamp algorithm for reconstructing a computer tomography density volume from x-ray images.

With cmake you can automatically create a Visual Studio project on Windows or a makefile on Linux.

============================================ Libraries ============================================

The required libraries are:

	1. OpenCV (currently 3.0)
			http://opencv.org/downloads.html
	3. Qt (currently 5.5)

============================================= Windows =============================================

	-	Qt can be downloaded precompiled as installer. QtCreator is _not_ required.
	-	OpenCV can also be downloaded precompiled

In cmake just set the necessary paths to the libraries and click configure and generate. The dlls should automatically be copied
to the Release and Debug directory.

============================================== Linux ==============================================

	-	OpenCV can be compiled using cmake and make
	-	Qt can be downloaded precompiled as an installer version

Then just use cmake as on Windows.