if (NOT DEFINED FFTW_ROOT_DIR)

	SET(FFTW_ROOT_DIR CACHE PATH "FFTW root directory")

endif(NOT DEFINED FFTW_ROOT_DIR)

if("${FFTW_ROOT_DIR}" STREQUAL "")

	message("Please specify the FFTW root directory (FFTW_ROOT_DIR) containing fftw3.h and libfftw3-3.lib for your machine and compiler.")

else("${FFTW_ROOT_DIR}" STREQUAL "")

	FIND_PATH(FFTW_INCLUDE_DIR 
	NAMES fftw3.h 
	HINTS ${FFTW_ROOT_DIR})

	if("${FFTW_INCLUDE_DIR}" STREQUAL "FFTW_INCLUDE_DIR-NOTFOUND")

		message("Header files could not be found in the directory you specified as FFTW root directory. Please choose again.")

	else("${FFTW_INCLUDE_DIR}" STREQUAL "FFTW_INCLUDE_DIR-NOTFOUND")

		SET(FFTW_INCLUDE_DIRS ${FFTW_INCLUDE_DIR})

	endif("${FFTW_INCLUDE_DIR}" STREQUAL "FFTW_INCLUDE_DIR-NOTFOUND")	

	FIND_LIBRARY(FFTW_LIB 
	NAMES libfftw3-3
	HINTS ${FFTW_ROOT_DIR})

	if(FFTW_LIB)

		list(APPEND FFTW_LIBRARIES ${FFTW_LIB})

	else(FFTW_LIB)

		message("ERROR: couldn't find the FFTW library libfftw3-3. Choose the correct root folder or try specifying the library paths manually.")

	endif(FFTW_LIB)

endif("${FFTW_ROOT_DIR}" STREQUAL "")