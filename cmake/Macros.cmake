########################################### Copy DLLS for passed modules ###########################################

macro(copydlls MODULELIST)
	foreach(ELEMENT ${${MODULELIST}})
		get_target_property(LOC_R ${ELEMENT} LOCATION_RELEASE)
		get_target_property(LOC_D ${ELEMENT} LOCATION_DEBUG)
		file(COPY ${LOC_R} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
		file(COPY ${LOC_D} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)

		get_target_property(DEPENDENCIES_RELEASE ${ELEMENT} IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE)
		foreach(DEPENDENCY ${DEPENDENCIES_RELEASE})
			if(TARGET ${DEPENDENCY})
				get_target_property(LOC_R ${DEPENDENCY} LOCATION_RELEASE)
				if(${LOC_R} MATCHES ".dll$")
					file(COPY ${LOC_R} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
				endif()
			endif()
		endforeach()	

		get_target_property(DEPENDENCIES_DEBUG ${ELEMENT} IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG)
		foreach(DEPENDENCY ${DEPENDENCIES_DEBUG})
			if(TARGET ${DEPENDENCY})
				get_target_property(LOC_D ${DEPENDENCY} LOCATION_DEBUG)
				if(${LOC_D} MATCHES ".dll$")
					file(COPY ${LOC_D} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
				endif()			
			endif()
		endforeach()	

		get_target_property(DEPENDENCIES_RELEASE ${ELEMENT} IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE)
		foreach(DEPENDENCY ${DEPENDENCIES_RELEASE})
			if(TARGET ${DEPENDENCY})
				get_target_property(LOC_R ${DEPENDENCY} LOCATION_RELEASE)
				if(${LOC_R} MATCHES ".dll$")
					file(COPY ${LOC_R} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
				endif()
			endif()
		endforeach()	

		get_target_property(DEPENDENCIES_DEBUG ${ELEMENT} IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG)
		foreach(DEPENDENCY ${DEPENDENCIES_DEBUG})
			if(TARGET ${DEPENDENCY})
				get_target_property(LOC_D ${DEPENDENCY} LOCATION_DEBUG)
				if(${LOC_D} MATCHES ".dll$")
					file(COPY ${LOC_D} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)			
				endif()
			endif()
		endforeach()

	endforeach()
endmacro()

########################################### Hides the specified variable from the gui ###########################################

macro(hide_from_gui VARIABLE)
	set(${VARIABLE} ${${VARIABLE}} CACHE INTERNAL "hidden" FORCE)
endmacro()

########################################### Asks for a path to the corresponding library ###########################################

# sets ${LIBNAME}_ROOT_FOUND (all uppercase) true if the desired file was found in the selected directory
# PATH_${LIBNAME}_ROOT (all uppercase) contains the selected path

macro (ask_for_path LIBNAME WINDOWS_DEFAULT_PATH LINUX_DEFAULT_PATH FILE_SEARCHED PATH_HINTS)
	string(TOUPPER ${LIBNAME} UPPERLIBNAME)
	if (NOT DEFINED PATH_${UPPERLIBNAME}_ROOT)

		if(WIN32)
			set(PATH_${UPPERLIBNAME}_ROOT ${WINDOWS_DEFAULT_PATH} CACHE PATH "${LIBNAME} root directory")
		elseif(UNIX)
			set(PATH_${UPPERLIBNAME}_ROOT ${LINUX_DEFAULT_PATH} CACHE PATH "${LIBNAME} root directory")
		endif(WIN32)

	endif()

	if("${PATH_${UPPERLIBNAME}_ROOT}" STREQUAL "")

		message("Please specify the ${LIBNAME} root directory (PATH_${UPPERLIBNAME}_ROOT).")

	else()

		set(REPLACED_PATH_HINTS)

		foreach(ELEMENT ${${PATH_HINTS}})
			if(NOT IS_ABSOLUTE path)
				set(ELEMENT ${PATH_${UPPERLIBNAME}_ROOT}/${ELEMENT})
				file(TO_CMAKE_PATH ${ELEMENT} ELEMENT)
				get_filename_component(ELEMENT ${ELEMENT} ABSOLUTE)
				list(APPEND REPLACED_PATH_HINTS ${ELEMENT})
			endif()
		endforeach()
		
		unset(${UPPERLIBNAME}_TMP CACHE)
		
		find_path(${UPPERLIBNAME}_TMP 
		NAMES ${${FILE_SEARCHED}}
		HINTS ${REPLACED_PATH_HINTS})	

		hide_from_gui(${UPPERLIBNAME}_TMP)

		if(${${UPPERLIBNAME}_TMP} STREQUAL "${UPPERLIBNAME}_TMP-NOTFOUND")

			message("The path you specified as ${LIBNAME} root directory (PATH_${UPPERLIBNAME}_ROOT) seems to be incorrect. Please chosse again.")

		else()

			set(PATH_${UPPERLIBNAME}_ROOT ${${UPPERLIBNAME}_TMP} CACHE PATH "${LIBNAME} root directory" FORCE)
			set(${UPPERLIBNAME}_ROOT_FOUND ON)
			hide_from_gui(${UPPERLIBNAME}_ROOT_FOUND)

		endif()

	endif()
endmacro()