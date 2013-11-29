if (NOT DEFINED OpenCV_ROOT_DIR)

	SET(OpenCV_ROOT_DIR CACHE PATH "OpenCV root directory")

endif(NOT DEFINED OpenCV_ROOT_DIR)

if("${OpenCV_ROOT_DIR}" STREQUAL "")

	message("Please specify the OpenCV root directory (OpenCV_ROOT_DIR).")

else("${OpenCV_ROOT_DIR}" STREQUAL "")

	set(OpenCV_MODULE_NAMES opencv_core opencv_highgui opencv_imgproc)

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

	foreach(element ${OpenCV_MODULE_NAMES})

		FIND_LIBRARY(${element}_MODULE_RELEASE 
		NAMES "${element}${OpenCV_VERSION}"
		HINTS ${LIBRARY_PATH_HINTS})

		FIND_LIBRARY(${element}_MODULE_DEBUG 
		NAMES "${element}${OpenCV_VERSION}d"
		HINTS ${LIBRARY_PATH_HINTS})

		if(${element}_MODULE_RELEASE)

			list(APPEND OpenCV_LIBRARIES_RELEASE ${${element}_MODULE_RELEASE})

		else(${element}_MODULE_RELEASE)

			message("ERROR: couldn't find the OpenCV release library ${element}. Choose the correct root folder or try specifying the library paths manually.")

		endif(${element}_MODULE_RELEASE)

		if(${element}_MODULE_DEBUG)

			list(APPEND OpenCV_LIBRARIES_DEBUG ${${element}_MODULE_DEBUG})

		else(${element}_MODULE_DEBUG)

			message("ERROR: couldn't find the OpenCV debug library ${element}. Choose the correct root folder or try specifying the library paths manually.")

		endif(${element}_MODULE_DEBUG)

	endforeach(element ${OpenCV_MODULE_NAMES})

endif("${OpenCV_ROOT_DIR}" STREQUAL "")