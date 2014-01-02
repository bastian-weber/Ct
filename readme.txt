With cmake you can automatically create a Visual Studio project.

Currently only Windows and Visual Studio 2012/2013 is supported. At least I didn't test anything else. Maybe it'll work, maybe not.
If you want to use Visual Studio 2013, make sure also the 2012 version is still installed on your system, otherwise it won't work because
it currently needs the debug DLLs of VS2012. This will be until I can get some OpenCV libraries compiled with VS2013.

- What you need:

	1. These sourcefiles
	2. OpenCV (download here: http://opencv.org/downloads.html)
	3. Cmake (download here: http://www.cmake.org/cmake/resources/software.html)

- Extract opencv to a place on your computer where you wish to keep it. (Please do not alter the internal structure of the OpenCV folder)
- I assume your OpenCV version is 2.4.7. If so, nothing has to be done. Otherwise: 
	- If your OpenCV version is not 2.4.7 the makefile has to be modified, so it finds this version of the libraries. Open the cmakelists.txt file and look for the line "set(OpenCV_VERSION 247)". Enter your version number instead of 247 here.
- Start cmake (the cmake-gui)
- In the field that sais "Where is the source code" you enter the path of this file (being the path where the cmakelists.txt and the code files are located)
- In the filed that sais "Where to build the binaries" you enter the path where your VisualStudio project shall be created. I would propose the following structure:
	-	/CV_project/sourcecode
  In the folder "sourcecode" you put the repository (the codefiles, the cmakelists.txt etc) and the folder "CV_project", meaning the parent folder, you select as output folder. Of course the folder names are arbitrary.
- Click the button "Configure". You will be prompted to select a compiler. Select either VS11 or VS12 (which is VisualStudio 2012 or 2013). I would suggest using the x64 version (to create a 64bit project). But also 32bit
  will work.
- After this step you will be prompted to enter the OpenCV root directory. So click on the little plus before "Ungrouped Entries" and behind "OpenCV_ROOT_DIR" select the path where you put OpenCV.
- Click configure again. There shouldn't be any errors and "Everything fine. Ready to generate." should be displayed in the output window.
- Click generate.

Now a VisualStudio project has been created in the output folder. You can open it with Visual Studio. Here you have to set the startup project, before you can compile. So in the solution explorer right click the Cv-project and select
"Set as StartUp project". Now you can compile as Release or Debug. But when running the .exe you will see that it can't find the OpenCV DLLs. You have to copy those into the folders where the .exe files are located. So go to your openCV
folder, then go to /build, then go either to /x64 or /x86, depending on whether you have a x64 or a x86 project, then go to /vc11/bin. Here you choose the dlls

	- opencv_core247.dll
	- opencv_highgui247.dll
	- opencv_imgproc247.dll

and copy them to your Release directory. Then you pick the dlls

	- opencv_core247d.dll
	- opencv_highgui247d.dll
	- opencv_imgproc247d.dll

and copy them to your Debug directory (note that they are called the same, except for the 'd' at the end).

Now you should be able to compile and run the program in Release as well as in Debug mode.

When something on the code is changed, you don't have to run cmake again. Only if new files or libraries are added, then you have to run cmake again. Fortunately it stores all variables in the cache, so you just have to click "Configure" and "Generate" and your project will be updated.

If there are any problems, just ask me.