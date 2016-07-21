======================================================== CT ========================================================

The repository for this application as well as precompiled binaries and example data can be found here: 
	
	https://bitbucket.org/bastian_weber/ct/wiki/Home

IMPORTANT: on Windows you'll have to install the included Visual C++ 2013 redistributable first! 
(vcredist_x64_install_first.exe)

============================================ Use of graphical interface ============================================

Launching the program without command line parameters will open up the graphical interface. Run start.sh on Linux.

-	To cycle through the projections use the left and right arrow keys.
-	To scroll through the volume cross-sections use the up and down arrow keys. Press X, Y or Z to select the 
	corresponding cross section axis.
-	Alt + scroll wheel can be used in both cases to scroll in larger steps.

============================================= Use as command line tool =============================================

The command line options are:

	-i [filepath] 		Specifies the path to the input csv file containing information about the images and 
						parameters. Long: --input
	-o [filepath] 		Specifies the file where the output volume shall be saved. Long: --output
	-b 					Optional. Runs with background priority. Long: --background
	-f [option] 		Optional. Sets the preprocessing filter. Options are 'ramlak', 'shepplogan' and 'hann'.
						Long: --filter
						Default: ramlak
	-n 					Optional. Disables CUDA. Long: --nocuda.
	-d 0,1,..,n 		Optional. Sets the cuda devices that shall be used. Option is a list of device ids seperated
						by comma.
						Long: --cudadevices
						Default: 0
	-w x,y 				Specifies the coefficients for the GPU weights; both are floating point values.
						X is the multiprocessor coefficient and y the memory bandwidth coefficient. 
						Long: --weights
						Default: 1,1
	-d [number] 		Optional. Sets the amount of VRAM to spare in Mb. Option is a positive integer. 
						Long: --cudasparememory.
						Default: 200
	-e [option]			Optional. Sets the byte order of the output. Options are 'littleendian' and 'bigendian'.
						Long: --byteorder
						Default: littleendian
	-j [option]			Optional. Sets the index order of the output. Options are 'zfastest' and 'xfastest'.
						Long: --indexorder
						Default: zfastest
	--xmin 0..1 		Optional. The lower x bound of the volume part that will be reconstructed.
	--xmax 0..1 		Optional. The upper x bound of the volume part that will be reconstructed.
	--ymin 0..1 		Optional. The lower y bound of the volume part that will be reconstructed.
	--ymax 0..1 		Optional. The upper y bound of the volume part that will be reconstructed.
	--zmin 0..1 		Optional. The lower z bound of the volume part that will be reconstructed.
	--zmax 0..1 		Optional. The upper z bound of the volume part that will be reconstructed.
	-h 					Displays help. Long: --help

===================================================== CtViewer =====================================================

The CtViewer application is able to load the reconstructed volumes from disc and display them in the form of
cross-sections. To open a volume just drag it or the corresponding info file onto the CtViewer window. By
right-clicking the application window a context menu shows up that contains all possible commands and also shows
the corresponding keyboard shortcuts.

Controls:

-	To scroll through the volume cross-sections use the up and down arrow keys. Press X, Y or Z to select the 
	corresponding cross section axis.
-	Press L or G to change bewtween local and global normalisation
-	Alt + scroll wheel can be used to scroll in larger steps.
-	Double clicking the application window will switch to fullscreen display
-	Ctrl + O to open a volume
-	Ctrl + S to save the current cross section as image

===================================================== Contact ======================================================

If you wish to contact the author, please write to the e-mail address bastian.weber@uni-weimar.de.