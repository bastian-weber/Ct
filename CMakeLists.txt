cmake_minimum_required (VERSION 3.0.2)

#=======================================================================#

#Opencv modules
set(OPENCV_MODULES_CV "opencv_core" "opencv_imgcodecs" "opencv_highgui" "opencv_imgproc")
set(QT_MODULES_CV "Qt5::Core" "Qt5::Gui" "Qt5::Widgets")

#=======================================================================#

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(Ct)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB SOURCEFILES "*.h"  "*.hpp"  "*.cpp")

add_executable(Ct ${SOURCEFILES})

#set entry point in case windows subsystem is used and enable whole program optimization
if(MSVC)
	set_target_properties(Ct PROPERTIES LINK_FLAGS "/ENTRY:\"mainCRTStartup\"")
	set_target_properties(Ct PROPERTIES LINK_FLAGS_RELEASE "/LTCG")
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL")
endif(MSVC)

#gcc flags (C++11 and some Qt flag)
if(CMAKE_COMPILER_IS_GNUCC)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fPIC")
endif(CMAKE_COMPILER_IS_GNUCC)

#OpenCV
set(OpenCV_STATIC OFF)
set(OPENCV_MODULES_ALL)
list(APPEND OPENCV_MODULES_ALL ${OPENCV_MODULES_CV})
list(REMOVE_DUPLICATES OPENCV_MODULES_ALL)
if(NOT UNIX)
	include(AskForOpenCVPath)
endif(NOT UNIX)
if(OPENCV_ROOT_FOUND OR UNIX)
	set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${OPENCV_ROOT_DIR})
	find_package(OpenCV 3.0 REQUIRED)
endif(OPENCV_ROOT_FOUND OR UNIX)

#Qt
set(QT_MODULES_ALL)
list(APPEND QT_MODULES_ALL ${QT_MODULES_CV})
list(REMOVE_DUPLICATES QT_MODULES_ALL)
include(AskForQtPath)
if(QT_ROOT_FOUND)
	set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${QT_ROOT_DIR})
	foreach(module ${QT_MODULES_ALL})
		string(REGEX REPLACE "::" "" MODULE_NAME ${module})
		find_package(${MODULE_NAME} 5.4.1 REQUIRED)
	endforeach(module ${QT_MODULES_ALL})
endif(QT_ROOT_FOUND)

#enable openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


#add lib files and copy dlls
include(Macros)

########################################### OpenCV Libs ###########################################

if((DEFINED OPENCV_ROOT_FOUND) OR UNIX)
	if(MSVC)
		message("Copying dlls, please wait.")
	endif()
	target_link_libraries(Ct ${OPENCV_MODULES_CV})
	#somehow include directories have to be added manually since OpenCV 3
	include_directories(${OpenCV_INCLUDE_DIRS})
	if(MSVC)
		copydlls("${OPENCV_MODULES_ALL}")
	endif(MSVC)
endif()

########################################### Qt Libs ###########################################

IF(DEFINED QT_ROOT_FOUND)

	target_link_libraries(Ct ${QT_MODULES_CV})

	#copy the dlls
	IF(MSVC)

		copydlls("${QT_MODULES_ALL}")	

		file(COPY ${QT_ROOT_DIR}/bin/icudt53.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
		file(COPY ${QT_ROOT_DIR}/bin/icuin53.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
		file(COPY ${QT_ROOT_DIR}/bin/icuuc53.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
		file(COPY ${QT_ROOT_DIR}/plugins/platforms DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)

		file(COPY ${QT_ROOT_DIR}/bin/icudt53.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
		file(COPY ${QT_ROOT_DIR}/bin/icuin53.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
		file(COPY ${QT_ROOT_DIR}/bin/icuuc53.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
		file(COPY ${QT_ROOT_DIR}/plugins/platforms DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)

	ENDIF(MSVC)

ENDIF(DEFINED QT_ROOT_FOUND)

IF((OPENCV_ROOT_FOUND OR UNIX) AND ((DEFINED FFTW_ROOT_FOUND) OR UNIX) AND QT_ROOT_FOUND)
	message("Everything fine, ready to generate.")
ENDIF()