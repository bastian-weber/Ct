if (NOT DEFINED OPENCV_ROOT_DIR)

	SET(OPENCV_ROOT_DIR CACHE PATH "OpenCV root directory")

endif(NOT DEFINED OPENCV_ROOT_DIR)

if("${OPENCV_ROOT_DIR}" STREQUAL "")

	message("Please specify the OpenCV root directory (OPENCV_ROOT_DIR).")

else("${OPENCV_ROOT_DIR}" STREQUAL "")

	FIND_PATH(OPENCV_TMP 
	NAMES OpenCVConfig.cmake
	HINTS ${OPENCV_ROOT_DIR} "${OPENCV_ROOT_DIR}" "${OPENCV_ROOT_DIR}/build" "${OPENCV_ROOT_DIR}/.." "${OPENCV_ROOT_DIR}/../.." "${OPENCV_ROOT_DIR}/../../..")	

	MARK_AS_ADVANCED(OPENCV_TMP)

	if(${OPENCV_TMP} STREQUAL "OPENCV_TMP-NOTFOUND")

		message("The path you specified as OpenCV root directory seems to be incorrect. Please chosse again.")

	else(${OPENCV_TMP} STREQUAL "OPENCV_TMP-NOTFOUND")

		set(OPENCV_ROOT_DIR ${OPENCV_TMP})
		set(OPENCV_ROOT_FOUND ON)

	endif(${OPENCV_TMP} STREQUAL "OPENCV_TMP-NOTFOUND")

endif("${OPENCV_ROOT_DIR}" STREQUAL "")