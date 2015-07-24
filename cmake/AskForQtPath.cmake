if (NOT DEFINED PATH_QT_ROOT)

	if(WIN32)
		set(PATH_QT_ROOT CACHE PATH "Qt root directory")
	elseif(UNIX)
		set(PATH_QT_ROOT "~/Qt5.5.0/5.5/gcc_64" CACHE PATH "Qt root directory")
	endif(WIN32)

endif(NOT DEFINED PATH_QT_ROOT)

if("${PATH_QT_ROOT}" STREQUAL "")

	message("Please specify the Qt root directory (PATH_QT_ROOT).")

else("${PATH_QT_ROOT}" STREQUAL "")

	unset(QT_TMP CACHE)

	find_path(QT_TMP 
	NAMES "lib/cmake/Qt5/Qt5Config.cmake"
	HINTS ${PATH_QT_ROOT} "${PATH_QT_ROOT}/../../.." "${PATH_QT_ROOT}/../.." "${PATH_QT_ROOT}/..")	

	#set(QT_TMP ${QT_TMP} CACHE INTERNAL "hidden" FORCE)
	hide_from_gui(QT_TMP)

	if(${QT_TMP} STREQUAL "QT_TMP-NOTFOUND")

		message("The path you specified as Qt root directory seems to be incorrect. Please chosse again.")

	else(${QT_TMP} STREQUAL "QT_TMP-NOTFOUND")

		set(PATH_QT_ROOT ${QT_TMP} CACHE PATH "Qt root directory" FORCE)
		set(QT_ROOT_FOUND ON)
		hide_from_gui(QT_ROOT_FOUND)

	endif(${QT_TMP} STREQUAL "QT_TMP-NOTFOUND")

endif("${PATH_QT_ROOT}" STREQUAL "")