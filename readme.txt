This project implements the Feldkamp algorithm for reconstructing a computer tomography density volume from x-ray images.

With cmake you can automatically create a Visual Studio project on Windows or a makefile on Linux.

============================================ Libraries ============================================

The required libraries are:

	1. OpenCV (currently 3.1)
			http://opencv.org/downloads.html
	2. NVIDIA CUDA Toolkit (currently 7.5)
			https://developer.nvidia.com/cuda-toolkit
	3. Qt (currently 5.6)
			https://www.qt.io/download-open-source/#section-2

============================================= Windows =============================================

	-	Install the Nidia Cuda Toolkit via the installer provided on the website 
	-	OpenCV can be compiled using cmake
		- 	Open cmake and specify the source and the build directories
		- 	Hit configure
		- 	Make sure to check the WITH_CUDA option in cmake (is default). Depending on which feature architectures you want to
		  	embed binary code for (e.g. Maxwell) you can add version numbers to the variable CUDA_ARCH_BIN in cmake (e.g. 5.2 for Maxwell)
		- 	Make sure the the cuda modules are enabled (BUILD_opencv_cudaarithm etc.)
		- 	The "world" modules currently might cause problems when building, so leave BUILD_opencv_world and
			BUILD_opencv_contrib_world unchecked
		- 	You can uncheck BUILD_TESTS, BUILD_DOCS and BUILD_PERF_TESTS to save time (won't build the unit tests, documentation and
			performance tests)
		- 	Set CMAKE_INSTALL_PREFIX to the path where the result of the build shall be installed.
		- 	Hit configure
		- 	Hit generate
		- 	Open the generated project in Visual Studio
		- 	Select Build > Batch build in the menu and check "Debug" and "Release" for the INSTALL target, then hit "Build"
		- 	When finished building the result will be in the install folder you specified
	-	Qt can be downloaded precompiled as installer. QtCreator is _not_ required.		

In cmake just set the necessary paths to the libraries and click configure and generate. The dlls should automatically be copied
to the Release and Debug directory.

============================================== Linux ==============================================

	-	Install Nvidia Cuda Toolkit
		-	The Nvidia driver can cause problems on some systems (especially if there is no Nvidia card). It will,
			however, be automatically installed if you install cuda via the deb package. So the safest way is to
			use the runfile-installer which gives you the option to not install the driver
		-	The driver is not required for compiling the project, only for being able to actually use the CUDA functionality.
			You can only do this if you actually have an Nvidia card. Building should also be possible without one.
	-	OpenCV can be compiled using cmake and make
		- 	Open cmake and specify the source and the build directories
		- 	Hit configure
		- 	Make sure to check the WITH_CUDA option in cmake (is default). Depending on which feature architectures you want to
		  	embed binary code for (e.g. Maxwell) you can add version numbers to the variable CUDA_ARCH_BIN in cmake (e.g. 5.2 for Maxwell)
		- 	Make sure the the cuda modules are enabled (BUILD_opencv_cudaarithm etc.)
		- 	The "world" modules currently might cause problems when building, so leave BUILD_opencv_world and
			BUILD_opencv_contrib_world unchecked
		- 	You can uncheck BUILD_TESTS, BUILD_DOCS and BUILD_PERF_TESTS to save time (won't build the unit tests, documentation and
			performance tests)
		- 	Set CMAKE_INSTALL_PREFIX to the path where the result of the build shall be installed. It is recommended to not use /usr/local
		  	if you want to have multiple different versions of OpenCV on your machine. Just pick some directory of your choice
		- 	Hit configure
		- 	Hit generate
		- 	Go to the build folder and run
			-	make
			-	make install
				-	if you chose an install folder that requires root permissions then run "sudo make install", however this
					is not recommended because it makes some files in the opencv build folder owned by root, which can then lead
					to other problems.
		- 	Select Build > Batch build in the menu and check "Debug" and "Release" for the INSTALL target, then hit "Build"
	-	Qt can be downloaded precompiled as an installer version
		-	If you don't have GL installed yet, run
			-	sudo apt-get install libgl1-mesa-dev

Then just use cmake as on Windows.