======================================================== CT ========================================================

The repository for this application as well as precompiled binaries and example data can be found here: 
	
	https://bitbucket.org/bastian_weber/ct/wiki/Home

============================================ Use of graphical interface ============================================

Launching the program without command line parameters will open up the graphical interface. Run start.sh on Linux.

-	To cycle through the projections use the left and right arrow keys.
-	To scroll through the volume cross-sections use the up and down arrow keys. Press X, Y or Z to select the 
	corresponding cross section axis.
-	Ctrl + scroll wheel can be used in both cases to scroll in larger steps.

============================================= Use as command line tool =============================================

The command line options are:

	-i [filepath] 		Specifies the path to the input csv file containing information about the images and parameters. Long: --input
	-o [filepath] 		Specifies the file where the output volume shall be saved. Long: --output
	-b 					Optional. Runs with background priority. Long: --background
	-f [option] 		Optional. Sets the preprocessing filter. Options are 'ramlak', 'shepplogan' and 'hann'. Long: --filter
	--xmin [0..1] 		Optional. The lower x bound of the volume part that will be reconstructed.
	--xmax [0..1] 		Optional. The upper x bound of the volume part that will be reconstructed.
	--ymin [0..1] 		Optional. The lower y bound of the volume part that will be reconstructed.
	--ymax [0..1] 		Optional. The upper y bound of the volume part that will be reconstructed.
	--zmin [0..1] 		Optional. The lower z bound of the volume part that will be reconstructed.
	--zmax [0..1] 		Optional. The upper z bound of the volume part that will be reconstructed.
	-h 					Displays help. Long: --help