		if(WIN32)
			set(FFTW_LIB "libfftw3f-3.lib")
		elseif(UNIX)
			set(FFTW_LIB "libfftw3.a")
		endif()

		FIND_PATH(FFTW_INCLUDE_DIR 
		NAMES fftw3.h)
			
		FIND_PATH(FFTW_LIB_DIR 
		NAMES ${FFTW_LIB})

		if(NOT ${FFTW_INCLUDE_DIR} STREQUAL "FFTW_INCLUDE_DIR-NOTFOUND" AND NOT ${FFTW_LIB_DIR} STREQUAL "FFTW_LIB_DIR-NOTFOUND")

			add_library(fftw SHARED IMPORTED)
			set_property(TARGET fftw APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
			set_target_properties(fftw PROPERTIES
			  IMPORTED_IMPLIB_RELEASE "${FFTW_LIB_DIR}/${FFTW_LIB}"
			  )

			set_property(TARGET fftw APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
			set_target_properties(fftw PROPERTIES
			  IMPORTED_IMPLIB_DEBUG "${FFTW_LIB_DIR}/${FFTW_LIB}"
			  )

			set_property(TARGET fftw PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FFTW_INCLUDE_DIR})

			if(WIN32)
				FIND_PATH(FFTW_DLL_DIR 
				NAMES libfftw3f-3.dll)

				if(NOT ${FFTW_DLL_DIR} STREQUAL "FFTW_DLL_DIR-NOTFOUND")
					set_target_properties(fftw PROPERTIES 
					IMPORTED_LOCATION_RELEASE "${FFTW_DLL_DIR}/libfftw3f-3.dll"
					IMPORTED_LOCATION_DEBUG "${FFTW_DLL_DIR}/libfftw3f-3.dll")
				else()
					message("FFTW dll could not be found.")
				endif()
			endif()

		else()

			message("FFTW could not be found.")

		endif()