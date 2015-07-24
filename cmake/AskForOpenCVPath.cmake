if (NOT DEFINED PATH_OPENCV_ROOT)

	set(PATH_OPENCV_ROOT CACHE PATH "OpenCV root directory")

endif(NOT DEFINED PATH_OPENCV_ROOT)

if("${PATH_OPENCV_ROOT}" STREQUAL "")

	message("Please specify the OpenCV root directory (PATH_OPENCV_ROOT).")

else("${PATH_OPENCV_ROOT}" STREQUAL "")

	unset(OPENCV_TMP CACHE)

	find_path(OPENCV_TMP 
	NAMES OpenCVConfig.cmake
	HINTS "${PATH_OPENCV_ROOT}" "${PATH_OPENCV_ROOT}/build" "${PATH_OPENCV_ROOT}/.." "${PATH_OPENCV_ROOT}/../.." "${PATH_OPENCV_ROOT}/../../..")	

	#hide_from_gui(OPENCV_TMP)

	if(${OPENCV_TMP} STREQUAL "OPENCV_TMP-NOTFOUND")

		message("The path you specified as OpenCV root directory seems to be incorrect. Please chosse again.")

	else(${OPENCV_TMP} STREQUAL "OPENCV_TMP-NOTFOUND")

		set(PATH_OPENCV_ROOT ${OPENCV_TMP} CACHE PATH "OpenCV root directory" FORCE)
		set(OPENCV_ROOT_FOUND ON)
		hide_from_gui(OPENCV_ROOT_FOUND)

	endif(${OPENCV_TMP} STREQUAL "OPENCV_TMP-NOTFOUND")

endif("${PATH_OPENCV_ROOT}" STREQUAL "")