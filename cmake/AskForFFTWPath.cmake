if (NOT DEFINED FFTW_ROOT_DIR)

	SET(FFTW_ROOT_DIR CACHE PATH "FFTW root directory")

endif(NOT DEFINED FFTW_ROOT_DIR)

if("${FFTW_ROOT_DIR}" STREQUAL "")

	message("Please specify the FFTW root directory (FFTW_ROOT_DIR).")

else("${FFTW_ROOT_DIR}" STREQUAL "")

	FIND_PATH(FFTW_TMP 
	NAMES fftw3.h libfftw3f-3.lib libfftw3f-3.dll
	HINTS ${FFTW_ROOT_DIR} "${FFTW_ROOT_DIR}")	

	MARK_AS_ADVANCED(FFTW_TMP)

	if(${FFTW_TMP} STREQUAL "FFTW_TMP-NOTFOUND")

		message("The path you specified as FFTW root directory seems to be incorrect. Please chosse again.")

	else(${FFTW_TMP} STREQUAL "FFTW_TMP-NOTFOUND")

		set(FFTW_ROOT_DIR ${FFTW_TMP})
		set(FFTW_ROOT_FOUND ON)

	endif(${FFTW_TMP} STREQUAL "FFTW_TMP-NOTFOUND")

endif("${FFTW_ROOT_DIR}" STREQUAL "")