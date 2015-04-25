		if(WIN32)
			set(FFTW_LIB_NAME "libfftw3f-3.dll")
			SET(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
		elseif(UNIX)
			set(FFTW_LIB_NAME "libfftw3.a")
		endif()

		FIND_PATH(FFTW_INCLUDE_DIR fftw3.h)
			
		FIND_LIBRARY(FFTW_LIB ${FFTW_LIB_NAME})

		if(NOT ${FFTW_INCLUDE_DIR} STREQUAL "FFTW_INCLUDE_DIR-NOTFOUND" AND NOT ${FFTW_LIB} STREQUAL "FFTW_LIB-NOTFOUND")

			add_library(fftw SHARED IMPORTED)
			set_property(TARGET fftw APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
			set_property(TARGET fftw APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)		
			set_target_properties(fftw PROPERTIES
								  INTERFACE_INCLUDE_DIRECTORIES ${FFTW_INCLUDE_DIR}
								  IMPORTED_LOCATION_RELEASE "${FFTW_LIB}"
								  IMPORTED_LOCATION_DEBUG "${FFTW_LIB}")

			if(WIN32)
				FIND_LIBRARY(FFTW_IMPLIB libfftw3f-3.lib)

				if(NOT ${FFTW_IMPLIB} STREQUAL "FFTW_IMPLIB-NOTFOUND")
					set_target_properties(fftw PROPERTIES
					  					  IMPORTED_IMPLIB_RELEASE "${FFTW_IMPLIB}"
					  					  IMPORTED_IMPLIB_DEBUG "${FFTW_IMPLIB}")
				else()
					message("FFTW dll could not be found.")
				endif()

			endif()

		else()

			message("FFTW could not be found.")

		endif()