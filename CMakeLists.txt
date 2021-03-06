cmake_minimum_required (VERSION 3.5)

#=======================================================================#

#Opencv modules
set(OPENCV_MODULES_CT "opencv_core" "opencv_highgui" "opencv_imgproc" "opencv_imgcodecs" "opencv_cudaarithm")
set(OPENCV_MODULES_CTVIEWER "opencv_core" "opencv_imgcodecs" "opencv_highgui" "opencv_imgproc")
#Qt modules
set(QT_MODULES_CT "Qt5::Core" "Qt5::Gui" "Qt5::Widgets")
set(QT_MODULES_CTVIEWER "Qt5::Core" "Qt5::Gui" "Qt5::Widgets")
#these modules will only be linked on windows
set(QT_MODULES_CT_WIN "Qt5::WinExtras")
set(QT_MODULES_CTVIEWER_WIN "Qt5::WinExtras")

set(PROJECT_NAME Ct)
set(EXECUTABLE_NAME_1 "Ct")
set(EXECUTABLE_NAME_2 "CtViewer")

#=======================================================================#

set(TARGETS "${EXECUTABLE_NAME_1}" "${EXECUTABLE_NAME_2}")

if(WIN32)
	list(APPEND QT_MODULES_CT ${QT_MODULES_CT_WIN})
	list(APPEND QT_MODULES_CTVIEWER ${QT_MODULES_CTVIEWER_WIN})
endif()

if(WIN32)
	set(COPY_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/Release")
	set(COPY_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/Debug")
else()
	#set exe output path
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/build")
	set(COPY_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
	set(COPY_DIRECTORY_DEBUG "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(Macros)

project(${PROJECT_NAME})

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

file(GLOB SOURCEFILES_CT "./Ct/*.h"  "./Ct/*.hpp"  "./Ct/*.cpp"  "./Ct/*.cu")
file(GLOB SOURCEFILES_CTVIEWER "./CtViewer/*.h"  "./CtViewer/*.hpp"  "./CtViewer/*.cpp"  "./CtViewer/*.cu")
file(GLOB SOURCEFILES_COMMON "./Common/*.h"  "./Common/*.hpp"  "./Common/*.cpp"  "./Common/*.cu")
list(APPEND SOURCEFILES_CT ${SOURCEFILES_COMMON})
list(APPEND SOURCEFILES_CTVIEWER ${SOURCEFILES_COMMON})
if(WIN32)
	#add application icon ressource
	list(APPEND SOURCEFILES_CT "${CMAKE_CURRENT_SOURCE_DIR}/Ct/ct.rc")
	list(APPEND SOURCEFILES_CTVIEWER "${CMAKE_CURRENT_SOURCE_DIR}/CtViewer/ctviewer.rc")
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

list(APPEND CUDA_NVCC_FLAGS "-use_fast_math")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

#OpenCV
set(OpenCV_STATIC OFF)
set(OPENCV_MODULES_ALL)
list(APPEND OPENCV_MODULES_ALL ${OPENCV_MODULES_CT} ${OPENCV_MODULES_CTVIEWER})
list(REMOVE_DUPLICATES OPENCV_MODULES_ALL)
set(OPENCV_FILES "OpenCVConfig.cmake")
set(OPENCV_PATH_HINTS "." "./build" "./.." "./../.." "./../../.." "./share/OpenCV")
ask_for_path("OpenCV" "" "/usr/local" OPENCV_FILES OPENCV_PATH_HINTS)
if(OPENCV_ROOT_FOUND)
	set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PATH_OPENCV_ROOT})
	find_package(OpenCV 3.1 REQUIRED)
endif()

if(OPENCV_ROOT_FOUND)
	#adds the opencv include dirs to nvcc
	CUDA_INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
endif()

#Qt
set(QT_MODULES_ALL)
list(APPEND QT_MODULES_ALL ${QT_MODULES_CT} ${QT_MODULES_CTVIEWER})
list(REMOVE_DUPLICATES QT_MODULES_ALL)
set(QT_FILES "lib/cmake/Qt5/Qt5Config.cmake")
set(QT_PATH_HINTS "." "./../../.." "./../.." "./..")
ask_for_path("Qt" "" "/opt/Qt5.6.0/5.6/gcc_64" QT_FILES QT_PATH_HINTS)
if(QT_ROOT_FOUND)
	set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PATH_QT_ROOT})
	foreach(module ${QT_MODULES_ALL})
		string(REGEX REPLACE "::" "" MODULE_NAME ${module})
		find_package(${MODULE_NAME} 5.6 REQUIRED)
	endforeach(module ${QT_MODULES_ALL})
endif(QT_ROOT_FOUND)

#enable openmp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

################################################ Configuration ################################################

cuda_add_executable(${EXECUTABLE_NAME_1} ${SOURCEFILES_CT})
cuda_add_cufft_to_target(${EXECUTABLE_NAME_1})
add_executable(${EXECUTABLE_NAME_2} ${SOURCEFILES_CTVIEWER})

include_directories("./Common")

#set entry point in case windows subsystem is used and enable whole program optimization
if(MSVC)
	set_target_properties(${EXECUTABLE_NAME_1} PROPERTIES LINK_FLAGS "/ENTRY:\"mainCRTStartup\"")
	set_target_properties(${EXECUTABLE_NAME_1} PROPERTIES LINK_FLAGS_RELEASE "/LTCG")
	set_target_properties(${EXECUTABLE_NAME_2} PROPERTIES LINK_FLAGS "/ENTRY:\"mainCRTStartup\"")
	set_target_properties(${EXECUTABLE_NAME_2} PROPERTIES LINK_FLAGS_RELEASE "/LTCG /SUBSYSTEM:WINDOWS")
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL")
endif(MSVC)

#gcc flags (C++11 and some Qt flag)
if(CMAKE_COMPILER_IS_GNUCC)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fPIC")
endif(CMAKE_COMPILER_IS_GNUCC)
if(UNIX)
	#make executable find libs in same folder
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath='$ORIGIN/lib'")
endif()

#add custom target for copying ressources
add_custom_target(COPY_RESSOURCES)
foreach(EXECUTABLE ${TARGETS})
	add_dependencies(${EXECUTABLE} COPY_RESSOURCES)
endforeach()

add_custom_target(COPY_LIBS)
foreach(EXECUTABLE ${TARGETS})
	add_dependencies(${EXECUTABLE} COPY_LIBS)
endforeach()

if(WIN32)
	add_custom_command(TARGET COPY_RESSOURCES PRE_BUILD 
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/deploy_ressources/general/data "${PROJECT_BINARY_DIR}/data"
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/deploy_ressources/general "${COPY_DIRECTORY_RELEASE}"
		COMMENT "Copying ressources to executable directory...")
elseif (UNIX)
	add_custom_command(TARGET COPY_RESSOURCES PRE_BUILD 
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/deploy_ressources/general "${COPY_DIRECTORY_RELEASE}"
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/deploy_ressources/linux "${COPY_DIRECTORY_RELEASE}"
		COMMENT "Copying ressources to executable directory...")
endif()



set(RELEASE_DLLS)
set(DEBUG_DLLS)

set(LIB_SUBFOLDER "")
set(PLATFORMS_SUBFOLDER "")
if(UNIX)
	set(LIB_SUBFOLDER "/lib")
	set(PLATFORMS_SUBFOLDER "/plugins")
endif()

#add lib files and copy libs
message("Looking for libs, please wait.")

########################################### OpenCV Libs ###########################################

if(DEFINED OPENCV_ROOT_FOUND)
	target_link_libraries(${EXECUTABLE_NAME_1} ${OPENCV_MODULES_CT})
	target_link_libraries(${EXECUTABLE_NAME_2} ${OPENCV_MODULES_CTVIEWER})
	#somehow include directories have to be added manually since OpenCV 3
	include_directories(${OpenCV_INCLUDE_DIRS})

	copydlls(RELEASE_DLLS DEBUG_DLLS OPENCV_MODULES_ALL)
	
	#copy cuda dlls for opencv
	if(${CUDA_FOUND} AND DEFINED CUDA_TOOLKIT_ROOT_DIR)
		if(MSVC)
			#only 64bit dlls are copied, might not work with 32bit project

			list(APPEND RELEASE_DLLS ${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll)
			list(APPEND RELEASE_DLLS ${CUDA_TOOLKIT_ROOT_DIR}/bin/cufft64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll)
			list(APPEND RELEASE_DLLS ${CUDA_TOOLKIT_ROOT_DIR}/bin/nppc64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll)
			list(APPEND RELEASE_DLLS ${CUDA_TOOLKIT_ROOT_DIR}/bin/nppi64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll)

			list(APPEND DEBUG_DLLS ${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll)
			list(APPEND DEBUG_DLLS ${CUDA_TOOLKIT_ROOT_DIR}/bin/cufft64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll)
			list(APPEND DEBUG_DLLS ${CUDA_TOOLKIT_ROOT_DIR}/bin/nppc64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll)
			list(APPEND DEBUG_DLLS ${CUDA_TOOLKIT_ROOT_DIR}/bin/nppi64_${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}.dll)
		elseif(UNIX)
			list(APPEND RELEASE_DLLS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so.7.5)
			list(APPEND RELEASE_DLLS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcufft.so.7.5)
			list(APPEND RELEASE_DLLS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppc.so.7.5)
			list(APPEND RELEASE_DLLS ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppi.so.7.5)
		endif()	
	endif()

	IF(UNIX)
		#some additional required libs
		list(APPEND RELEASE_DLLS /usr/lib/x86_64-linux-gnu/libgomp.so.1)
		list(APPEND RELEASE_DLLS /usr/lib/x86_64-linux-gnu/libstdc++.so.6)


	ENDIF()
endif()

########################################### Qt Libs ###########################################

IF(DEFINED QT_ROOT_FOUND)

	target_link_libraries(${EXECUTABLE_NAME_1} ${QT_MODULES_CT} )
	target_link_libraries(${EXECUTABLE_NAME_2} ${QT_MODULES_CTVIEWER})

	#copy libs
	copydlls(RELEASE_DLLS DEBUG_DLLS QT_MODULES_ALL)

	IF(MSVC)
		list(APPEND RELEASE_DLLS ${PATH_QT_ROOT}/bin/icudt54.dll)
		list(APPEND RELEASE_DLLS ${PATH_QT_ROOT}/bin/icuin54.dll)
		list(APPEND RELEASE_DLLS ${PATH_QT_ROOT}/bin/icuuc54.dll)
		
		list(APPEND DEBUG_DLLS ${PATH_QT_ROOT}/bin/icudt54.dll)
		list(APPEND DEBUG_DLLS ${PATH_QT_ROOT}/bin/icuin54.dll)
		list(APPEND DEBUG_DLLS ${PATH_QT_ROOT}/bin/icuuc54.dll)
	ELSEIF(UNIX)
		list(APPEND RELEASE_DLLS ${PATH_QT_ROOT}/lib/libicudata.so.56)
		list(APPEND RELEASE_DLLS ${PATH_QT_ROOT}/lib/libicui18n.so.56)
		list(APPEND RELEASE_DLLS ${PATH_QT_ROOT}/lib/libicuuc.so.56)
		list(APPEND RELEASE_DLLS ${PATH_QT_ROOT}/lib/libQt5DBus.so.5)
		list(APPEND RELEASE_DLLS ${PATH_QT_ROOT}/lib/libQt5XcbQpa.so.5)
	ENDIF(MSVC)

	add_custom_command(TARGET COPY_LIBS PRE_BUILD 
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${PATH_QT_ROOT}/plugins/platforms "${COPY_DIRECTORY_DEBUG}${PLATFORMS_SUBFOLDER}/platforms"
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${PATH_QT_ROOT}/plugins/platforms "${COPY_DIRECTORY_RELEASE}${PLATFORMS_SUBFOLDER}/platforms"
		COMMENT "Copying Qt platforms to executable directory...")

ENDIF(DEFINED QT_ROOT_FOUND)

#copy libs
if(DEFINED RELEASE_DLLS AND DEFINED DEBUG_DLLS)

	set(CREATE_LIB_DIR_COMMAND "-E" "make_directory" "${COPY_DIRECTORY_RELEASE}/lib")

	set(COPY_COMMAND_RELEASE "-E" "copy_if_different")
	set(COPY_COMMAND_DEBUG "-E" "copy_if_different")
	
	list(REMOVE_DUPLICATES RELEASE_DLLS)
	list(REMOVE_DUPLICATES DEBUG_DLLS)

	foreach(DLL ${RELEASE_DLLS})
		list(APPEND COPY_COMMAND_RELEASE "\"${DLL}\"")
	endforeach()
	foreach(DLL ${DEBUG_DLLS})
		list(APPEND COPY_COMMAND_DEBUG "\"${DLL}\"")
	endforeach()

	list(APPEND COPY_COMMAND_RELEASE "\"${COPY_DIRECTORY_RELEASE}${LIB_SUBFOLDER}\"")
	list(APPEND COPY_COMMAND_DEBUG "\"${COPY_DIRECTORY_DEBUG}${LIB_SUBFOLDER}\"")

	if(UNIX)
		add_custom_command(TARGET COPY_LIBS POST_BUILD COMMAND ${CMAKE_COMMAND} ${CREATE_LIB_DIR_COMMAND})	
	endif()
	foreach(EXECUTABLE ${TARGETS})
		add_custom_command(TARGET COPY_LIBS POST_BUILD COMMAND ${CMAKE_COMMAND} ${COPY_COMMAND_RELEASE})
		if(MSVC)
			add_custom_command(TARGET COPY_LIBS POST_BUILD COMMAND ${CMAKE_COMMAND} ${COPY_COMMAND_DEBUG})
		endif()
	endforeach()

endif()

IF(OPENCV_ROOT_FOUND AND QT_ROOT_FOUND AND CUDA_FOUND)
	message("Everything fine, ready to generate.")
ENDIF()
