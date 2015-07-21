if (NOT DEFINED PATH_QT_ROOT)

	if(WIN32)
		SET(PATH_QT_ROOT CACHE PATH "OpenCV root directory")
	elseif(UNIX)
		SET(PATH_QT_ROOT "~/Qt5.5.0/5.5/gcc_64" CACHE PATH "Qt root directory")
	endif(WIN32)

endif(NOT DEFINED PATH_QT_ROOT)

if("${PATH_QT_ROOT}" STREQUAL "")

	message("Please specify the Qt root directory (PATH_QT_ROOT).")

else("${PATH_QT_ROOT}" STREQUAL "")

	FIND_PATH(QT_TMP 
	NAMES "lib/cmake/Qt5/Qt5Config.cmake"
	HINTS ${PATH_QT_ROOT} "${PATH_QT_ROOT}/../../.." "${PATH_QT_ROOT}/../.." "${PATH_QT_ROOT}/..")	

	MARK_AS_ADVANCED(QT_TMP)

	if(${QT_TMP} STREQUAL "QT_TMP-NOTFOUND")

		message("The path you specified as Qt root directory seems to be incorrect. Please chosse again.")

	else(${QT_TMP} STREQUAL "QT_TMP-NOTFOUND")

		set(PATH_QT_ROOT ${QT_TMP})
		set(QT_ROOT_FOUND ON)

	endif(${QT_TMP} STREQUAL "QT_TMP-NOTFOUND")

endif("${PATH_QT_ROOT}" STREQUAL "")