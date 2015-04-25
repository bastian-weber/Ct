cmake_minimum_required (VERSION 2.6)


#Opencv modules
set(OPENCV_MODULES_CV "opencv_core" "opencv_highgui" "opencv_imgproc")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(Cv)

file(GLOB SOURCEFILES "*.h"  "*.hpp"  "*.cpp")

add_executable(Cv ${SOURCEFILES})

#set entry point in case windows subsystem is used and enable whole program optimization
if(MSVC)
	set_target_properties(Cv PROPERTIES LINK_FLAGS "/ENTRY:\"mainCRTStartup\"")
	set_target_properties(Cv PROPERTIES LINK_FLAGS_RELEASE "/LTCG")
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL")
endif(MSVC)

#gcc flags (C++11 and some Qt flag)
if(CMAKE_COMPILER_IS_GNUCC)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fPIC")
endif(CMAKE_COMPILER_IS_GNUCC)

#OpenCV
set(OPENCV_MODULES_ALL)
list(APPEND OPENCV_MODULES_ALL ${OPENCV_MODULES_CV})
list(REMOVE_DUPLICATES OPENCV_MODULES_ALL)
if(NOT UNIX)
	include(AskForOpenCVPath)
endif(NOT UNIX)
if(OPENCV_ROOT_FOUND OR UNIX)
	set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${OPENCV_ROOT_DIR})
	find_package(OpenCV 2.4.10 REQUIRED)
endif(OPENCV_ROOT_FOUND OR UNIX)

#FFTW
if(NOT UNIX)
	include(AskForFFTWPath)
endif(NOT UNIX)
if(FFTW_ROOT_FOUND OR UNIX)
	set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${FFTW_ROOT_DIR})	
	find_package(FFTW REQUIRED)
endif()

include(Macros)

if((DEFINED OPENCV_ROOT_FOUND) OR UNIX)

	if(MSVC)
		message("Copying dlls, please wait.")
	endif()

	target_link_libraries(Cv ${OPENCV_MODULES_CV})

	if(MSVC)
		copydlls("${OPENCV_MODULES_ALL}")
	endif(MSVC)

endif()

if((DEFINED FFTW_ROOT_FOUND) OR UNIX)

	target_link_libraries(Cv fftw)

	if(MSVC)
		copydlls("fftw")
	endif(MSVC)

endif()

IF((OPENCV_ROOT_FOUND OR UNIX) AND ((DEFINED FFTW_ROOT_FOUND) OR UNIX))
	message("Everything fine, ready to generate.")
ENDIF()