# Filtered Backprojection for X-Ray Cone-Beam Computed Tomography on CPU and GPU

## Open source implementation of the Feldkamp-Davis-Kress (FDK) algorithm in C++ and CUDA

### Download (1.41)

The Github [releases section](https://github.com/bastian-weber/Ct/releases) contains precompiled binaries for Windows (x64) and Linux (x64) as well as two small example data sets (sneakers scan (256x146), and toy figure scan (224x256)). The Visual C++ 2013 Redistributable is required on Windows.

A larger example data set can be downloaded from externally: [toy figure scan (875x1000)](https://drive.google.com/open?id=0B5U6xEpkBI6cT3BjUnc0OXNzQ3c)

In my master's thesis I provide a detailed explanation of the algorithm and the implementation. It is available for download as a PDF file from the [releases section](https://github.com/bastian-weber/Ct/releases) as well.

### CUDA

The reconstruction can either be run on the CPU or the GPU using CUDA if an Nvidia graphics card is available. The CUDA reconstruction is much faster but requires an Nvidia graphics card.

![Performance comparison][chart]

### Use of graphical interface

The program offers a graphical interface that opens up if it is launched without any command line parameters.

![Projections][screenshot1]

Via the graphical interface a config file path can be selected as well as the desired frequency filter. Also the bounds for the reconstructed volume can be set.

* To cycle through the projections use the left and right arrow keys.
* To scroll through the volume cross-sections use the up and down arrow keys. Press X, Y or Z to select the corresponding cross section axis.
* Alt + scroll wheel can be used in both cases to scroll in larger steps.

By choosing the corresponding option it is possible to generate a batch file that runs the reconstruction using the settings that have be made via the graphical user interface.

![Finished reconstruction][screenshot2]

### Use as a command line tool

It can also be used as a command line application. In this case all output will happen via the console and no graphical interface will show up.

The command line options are:

* __-i [filepath]__ Specifies the path to the input csv file containing information about the images and parameters. Long: --input
* __-o [filepath]__ Specifies the file where the output volume shall be saved. Long: --output
* __-b__ Optional. Runs with background priority. Long: --background
* __-f [option]__ Optional. Sets the preprocessing filter. Options are 'ramlak', 'shepplogan' and 'hann'. Long: --filter Default: ramlak
* __-n__ Optional. Disables CUDA. Long: --nocuda.
* __-c__ Optional. Selects the use of CPU processing when using CUDA for the reconstruction. Long: --cpuPreprocessing
* __-d 0,1,..,n__ Optional. Sets the cuda devices that shall be used. Option is a list of device ids seperated by comma. Long: --cudadevices. Default: 0
* __-w x,y__ Specifies the coefficients for the GPU weights; both are floating point values. X is the multiprocessor coefficient and y the memory bandwidth coefficient. Long: --weights Default: 1,1
* __-m [number]__ Optional. Sets the amount of VRAM to spare in Mb. Option is a positive integer. Long: --cudasparememory. Default: 200
* __-e [option]__ Optional. Sets the byte order of the output. Options are 'littleendian' and 'bigendian'. Long: --byteorder Default: littleendian
* __-j [option]__ Optional. Sets the index order of the output. Options are 'zfastest' and 'xfastest'. Long: --indexorder Default: zfastest
* __--xmin [0..1]__ Optional. The lower x bound of the volume part that will be reconstructed.
* __--xmax [0..1]__ Optional. The upper x bound of the volume part that will be reconstructed.
* __--ymin [0..1]__ Optional. The lower y bound of the volume part that will be reconstructed.
* __--ymax [0..1]__ Optional. The upper y bound of the volume part that will be reconstructed.
* __--zmin [0..1]__ Optional. The lower z bound of the volume part that will be reconstructed.
* __--zmax [0..1]__ Optional. The upper z bound of the volume part that will be reconstructed.
* __-h__ Displays help. Long: --help

### CtViewer

The CtViewer application is able to load the reconstructed volumes from disc and display them in the form of cross-sections. To open a volume just drag it or the corresponding info file onto the CtViewer window. By right-clicking the application window a context menu shows up that contains all possible commands and also shows the corresponding keyboard shortcuts.

Controls:

* To scroll through the volume cross-sections use the up and down arrow keys. Press X, Y or Z to select the corresponding cross section axis.
* Press L or G to change bewtween local and global normalisation
* Alt + scroll wheel can be used to scroll in larger steps.
* Double clicking the application window will switch to fullscreen display
* Ctrl + O to open a volume
* Ctrl + S to save the current cross section as image

### Contact

If you wish to contact the author, please write to the e-mail address bastian.home@gmail.com.

### Output example

![Finished reconstruction visualised using volume rendering][sneaker]

### Notes and Disclaimer

We do not guarantee correctness of the reconstruction results.

This software is provided by the copyright holders and contributors “as is” and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall copyright holders or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

[chart]: readme_images/chart.png
[screenshot1]: readme_images/screenshot1.png
[screenshot2]: readme_images/screenshot2.png
[sneaker]: readme_images/sneaker.jpg