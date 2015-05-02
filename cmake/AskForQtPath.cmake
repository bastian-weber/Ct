if (NOT DEFINED QT_ROOT_DIR)

	if(WIN32)
		SET(QT_ROOT_DIR CACHE PATH "OpenCV root directory")
	elseif(UNIX)
		SET(QT_ROOT_DIR "~/Qt5.4.0/5.4/gcc_64" CACHE PATH "Qt root directory")
	endif(WIN32)

endif(NOT DEFINED QT_ROOT_DIR)

if("${QT_ROOT_DIR}" STREQUAL "")

	message("Please specify the Qt root directory (QT_ROOT_DIR).")

else("${QT_ROOT_DIR}" STREQUAL "")

	FIND_PATH(QT_TMP 
	NAMES "lib/cmake/Qt5/Qt5Config.cmake"
	HINTS ${QT_ROOT_DIR} "${QT_ROOT_DIR}/../../.." "${QT_ROOT_DIR}/../.." "${QT_ROOT_DIR}/..")	

	MARK_AS_ADVANCED(QT_TMP)

	if(${QT_TMP} STREQUAL "QT_TMP-NOTFOUND")

		message("The path you specified as Qt root directory seems to be incorrect. Please chosse again.")

	else(${QT_TMP} STREQUAL "QT_TMP-NOTFOUND")

		set(QT_ROOT_DIR ${QT_TMP})
		set(QT_ROOT_FOUND ON)

	endif(${QT_TMP} STREQUAL "QT_TMP-NOTFOUND")

endif("${QT_ROOT_DIR}" STREQUAL "")