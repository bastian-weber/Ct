cmake_minimum_required (VERSION 3.0.2)

#=======================================================================#

#Opencv modules
set(OPENCV_MODULES_CV "opencv_core" "opencv_highgui" "opencv_imgproc" "opencv_imgcodecs" "opencv_cudaarithm" "opencv_cudafilters" "opencv_cudaimgproc" "opencv_cudawarping")
#Qt modules
set(QT_MODULES_CV "Qt5::Core" "Qt5::Gui" "Qt5::Widgets")
#these modules will only be linked on windows
set(QT_MODULES_CV_WIN "Qt5::WinExtras")

#=======================================================================#

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(Macros)

project(Ct)

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB SOURCEFILES "*.h"  "*.hpp"  "*.cpp"  "*.cu")

if(WIN32)
	#add application icon ressource
	list(APPEND SOURCEFILES "${CMAKE_CURRENT_SOURCE_DIR}/ct.rc")
endif()

#CUDA
#use dynamic linking
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA)
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=sm_20")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=sm_30")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_32,code=sm_32")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_35,code=sm_35")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_50,code=sm_50")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_52,code=sm_52")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_53,code=sm_53")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_53,code=compute_53")

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

#OpenCV
set(OpenCV_STATIC OFF)
set(OPENCV_MODULES_ALL)
list(APPEND OPENCV_MODULES_ALL ${OPENCV_MODULES_CV})
list(REMOVE_DUPLICATES OPENCV_MODULES_ALL)
set(OPENCV_FILES "OpenCVConfig.cmake")
set(OPENCV_PATH_HINTS "." "./build" "./.." "./../.." "./../../.." "./share/OpenCV")
ask_for_path("OpenCV" "" "/usr/local" OPENCV_FILES OPENCV_PATH_HINTS)
if(OPENCV_ROOT_FOUND)
	set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PATH_OPENCV_ROOT})
	find_package(OpenCV 3.0 REQUIRED)
endif()

if(OPENCV_ROOT_FOUND)
	#adds the opencv include dirs to nvcc
	CUDA_INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
endif()

#Qt
set(QT_MODULES_ALL)
list(APPEND QT_MODULES_ALL ${QT_MODULES_CV})
if(WIN32)
	list(APPEND QT_MODULES_ALL ${QT_MODULES_CV_WIN})
endif()
list(REMOVE_DUPLICATES QT_MODULES_ALL)
set(QT_FILES "lib/cmake/Qt5/Qt5Config.cmake")
set(QT_PATH_HINTS "." "./../../.." "./../.." "./..")
ask_for_path("Qt" "" "~/Qt/5.5/gcc_64" QT_FILES QT_PATH_HINTS)
if(QT_ROOT_FOUND)
	set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PATH_QT_ROOT})
	foreach(module ${QT_MODULES_ALL})
		string(REGEX REPLACE "::" "" MODULE_NAME ${module})
		find_package(${MODULE_NAME} 5.5 REQUIRED)
	endforeach(module ${QT_MODULES_ALL})
endif(QT_ROOT_FOUND)

#enable openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

################################################ Configuration ################################################

cuda_add_executable(Ct ${SOURCEFILES})

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

#add custom target for copying ressources
add_custom_target(COPY_RESSOURCES)
if(WIN32)
	add_custom_command(TARGET COPY_RESSOURCES POST_BUILD 
					   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/deploy_ressources/general/data ${PROJECT_BINARY_DIR}/data
					   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/deploy_ressources/general ${PROJECT_BINARY_DIR}/Release)
elseif (UNIX)
	add_custom_command(TARGET COPY_RESSOURCES POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/deploy_ressources/general ${PROJECT_BINARY_DIR}
					   TARGET COPY_RESSOURCES POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/deploy_ressources/linux ${PROJECT_BINARY_DIR})
endif()
add_dependencies(Ct COPY_RESSOURCES)

#add lib files and copy dlls

########################################### OpenCV Libs ###########################################

if(DEFINED OPENCV_ROOT_FOUND)
	if(MSVC)
		message("Copying dlls, please wait.")
	endif()
	target_link_libraries(Ct ${OPENCV_MODULES_CV})
	#somehow include directories have to be added manually since OpenCV 3
	include_directories(${OpenCV_INCLUDE_DIRS})
	if(MSVC)
		copydlls(OPENCV_MODULES_ALL)

		#copy cuda dlls for opencv
		if(${CUDA_FOUND} AND DEFINED CUDA_TOOLKIT_ROOT_DIR)
			#only 64bit dlls are copied, might not work with 32bit projects
			file(COPY ${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
			file(COPY ${CUDA_TOOLKIT_ROOT_DIR}/bin/cufft64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
			file(COPY ${CUDA_TOOLKIT_ROOT_DIR}/bin/nppc64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
			file(COPY ${CUDA_TOOLKIT_ROOT_DIR}/bin/nppi64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
			file(COPY ${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
			file(COPY ${CUDA_TOOLKIT_ROOT_DIR}/bin/cufft64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
			file(COPY ${CUDA_TOOLKIT_ROOT_DIR}/bin/nppc64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
			file(COPY ${CUDA_TOOLKIT_ROOT_DIR}/bin/nppi64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
		endif()	

	endif(MSVC)
endif()

########################################### Qt Libs ###########################################

IF(DEFINED QT_ROOT_FOUND)

	target_link_libraries(Ct ${QT_MODULES_ALL})

	#copy the dlls
	IF(MSVC)

		copydlls(QT_MODULES_ALL)	

		file(COPY ${PATH_QT_ROOT}/bin/icudt54.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
		file(COPY ${PATH_QT_ROOT}/bin/icuin54.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
		file(COPY ${PATH_QT_ROOT}/bin/icuuc54.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
		file(COPY ${PATH_QT_ROOT}/plugins/platforms DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)

		file(COPY ${PATH_QT_ROOT}/bin/icudt54.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
		file(COPY ${PATH_QT_ROOT}/bin/icuin54.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
		file(COPY ${PATH_QT_ROOT}/bin/icuuc54.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)
		file(COPY ${PATH_QT_ROOT}/plugins/platforms DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Release)

	ENDIF(MSVC)

ENDIF(DEFINED QT_ROOT_FOUND)

IF(OPENCV_ROOT_FOUND AND QT_ROOT_FOUND AND CUDA_FOUND)
	message("Everything fine, ready to generate.")
ENDIF()