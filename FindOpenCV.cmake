if (NOT DEFINED OpenCV_ROOT_DIR)

	SET(OpenCV_ROOT_DIR CACHE PATH "OpenCV root directory")

endif(NOT DEFINED OpenCV_ROOT_DIR)

if("${OpenCV_ROOT_DIR}" STREQUAL "")

	message("Please specify the OpenCV root directory (OpenCV_ROOT_DIR).")

else("${OpenCV_ROOT_DIR}" STREQUAL "")

	IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
		
		#This is 64bit
		set(LIBRARY_PATH_HINTS ${OpenCV_ROOT_DIR}/x64/vc11/lib ${OpenCV_ROOT_DIR}/build/x64/vc11/lib)

	ELSEIF(CMAKE_SIZEOF_VOID_P EQUAL 4)
		
		#This is 32bit
		set(LIBRARY_PATH_HINTS ${OpenCV_ROOT_DIR}/x86/vc11/lib ${OpenCV_ROOT_DIR}/build/x86/vc11/lib)		

	ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 8)

	FIND_PATH(OpenCV_INCLUDE_DIR 
	NAMES opencv2/core/core.hpp opencv2/highgui/highgui.hpp opencv2/imgproc/imgproc.hpp 
	HINTS ${OpenCV_ROOT_DIR}/include ${OpenCV_ROOT_DIR}/build/include)	

	if("${OpenCV_INCLUDE_DIR}" STREQUAL "OpenCV_INCLUDE_DIR-NOTFOUND")

		message("Header files could not be found in the directory you specified as OpenCV root directory. Please choose again.")

	else("${OpenCV_INCLUDE_DIR}" STREQUAL "OpenCV_INCLUDE_DIR-NOTFOUND")

		SET(OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIR})

	endif("${OpenCV_INCLUDE_DIR}" STREQUAL "OpenCV_INCLUDE_DIR-NOTFOUND")

	FIND_LIBRARY(OpenCV_CORE_RELEASE 
	NAMES "opencv_core${OpenCV_VERSION}"
	HINTS ${LIBRARY_PATH_HINTS})

	FIND_LIBRARY(OpenCV_CORE_DEBUG 
	NAMES "opencv_core${OpenCV_VERSION}d"
	HINTS ${LIBRARY_PATH_HINTS})

	FIND_LIBRARY(OpenCV_HIGHGUI_RELEASE 
	NAMES "opencv_highgui${OpenCV_VERSION}"
	HINTS ${LIBRARY_PATH_HINTS})

	FIND_LIBRARY(OpenCV_HIGHGUI_DEBUG 
	NAMES "opencv_highgui${OpenCV_VERSION}d"
	HINTS ${LIBRARY_PATH_HINTS})

	FIND_LIBRARY(OpenCV_IMGPROC_RELEASE 
	NAMES "opencv_imgproc${OpenCV_VERSION}"
	HINTS ${LIBRARY_PATH_HINTS})

	FIND_LIBRARY(OpenCV_IMGPROC_DEBUG 
	NAMES "opencv_imgproc${OpenCV_VERSION}d"
	HINTS ${LIBRARY_PATH_HINTS})	

	if(OpenCV_CORE_RELEASE AND OpenCV_CORE_DEBUG AND OpenCV_HIGHGUI_RELEASE AND OpenCV_HIGHGUI_DEBUG AND OpenCV_IMGPROC_RELEASE AND OpenCV_IMGPROC_DEBUG)

		SET(OpenCV_LIBRARIES_RELEASE ${OpenCV_CORE_RELEASE} 
									 ${OpenCV_HIGHGUI_RELEASE} 
									 ${OpenCV_IMGPROC_RELEASE})

		SET(OpenCV_LIBRARIES_DEBUG ${OpenCV_CORE_DEBUG} 
								   ${OpenCV_HIGHGUI_DEBUG} 
								   ${OpenCV_IMGPROC_DEBUG})	

	else(OpenCV_CORE_RELEASE AND OpenCV_CORE_DEBUG AND OpenCV_HIGHGUI_RELEASE AND OpenCV_HIGHGUI_DEBUG AND OpenCV_IMGPROC_RELEASE AND OpenCV_IMGPROC_DEBUG)

		message("Not all openCV library files could be found. Please correct your openCV directory. Otherwise you can also choose each libary file manually.")

	endif(OpenCV_CORE_RELEASE AND OpenCV_CORE_DEBUG AND OpenCV_HIGHGUI_RELEASE AND OpenCV_HIGHGUI_DEBUG AND OpenCV_IMGPROC_RELEASE AND OpenCV_IMGPROC_DEBUG)	

endif("${OpenCV_ROOT_DIR}" STREQUAL "")