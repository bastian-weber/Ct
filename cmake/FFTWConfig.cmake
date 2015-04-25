		if(WIN32)
			set(FFTW_LIB_NAME "libfftw3f-3.lib")
		elseif(UNIX)
			set(FFTW_LIB_NAME "libfftw3.a")
		endif()

		FIND_PATH(FFTW_INCLUDE_DIR fftw3.h)
			
		FIND_LIBRARY(FFTW_LIB ${FFTW_LIB_NAME})

		if(NOT ${FFTW_INCLUDE_DIR} STREQUAL "FFTW_INCLUDE_DIR-NOTFOUND" AND NOT ${FFTW_LIB} STREQUAL "FFTW_LIB-NOTFOUND")

			add_library(fftw SHARED IMPORTED)
			set_property(TARGET fftw APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
			set_target_properties(fftw PROPERTIES
			  IMPORTED_IMPLIB_RELEASE "${FFTW_LIB}"
			  )

			set_property(TARGET fftw APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
			set_target_properties(fftw PROPERTIES
			  IMPORTED_IMPLIB_DEBUG "${FFTW_LIB}"
			  )

			set_property(TARGET fftw PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FFTW_INCLUDE_DIR})

			if(WIN32)
				FIND_FILE(FFTW_DLL libfftw3f-3.dll)

				if(NOT ${FFTW_DLL} STREQUAL "FFTW_DLL-NOTFOUND")
					set_target_properties(fftw PROPERTIES 
					IMPORTED_LOCATION_RELEASE "${FFTW_DLL}"
					IMPORTED_LOCATION_DEBUG "${FFTW_DLL}")
				else()
					message("FFTW dll could not be found.")
				endif()
			endif()

		else()

			message("FFTW could not be found.")

		endif()