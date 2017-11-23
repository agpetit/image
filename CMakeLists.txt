include(${SOFA_CMAKE_DIR}/preProject.cmake)


set(SOURCE_FILES
    initImage.cpp
    ImageContainer.cpp
    ImageFilter.cpp
    ImageOperation.cpp
    ImageTransform.cpp
    TransferFunction.cpp
    ImageValuesFromPositions.cpp
    MergeImages.cpp
    ImageAccumulator.cpp
    DepthMapToMeshEngine.cpp
    MeshToImageEngine.cpp
    MarchingCubesEngine.cpp 
    ImageSampler.cpp
    ImageExporter.cpp 
    ImageDataDisplay.cpp
    VoronoiToMeshEngine.cpp
    #OpenniCam.cpp
    RealSenseCam.cpp
	#Openni2Cam.cpp

)

set(HEADER_FILES
    initImage.h
    ImageTypes.h
    ImageContainer.h
    ImageFilter.h
    ImageOperation.h
    ImageTransform.h
    TransferFunction.h
    ImageValuesFromPositions.h
    MergeImages.h
    ImageAccumulator.h
    DepthMapToMeshEngine.h
    MeshToImageEngine.h
    MarchingCubesEngine.h
    ImageSampler.h
    ImageExporter.h
    VectorVis.h 
    ImageAlgorithms.h 
    Containers.h 
    ImageDataDisplay.h
    VoronoiToMeshEngine.h
    #OpenniCam.h
    RealSenseCam.h
	#Openni2Cam.h

)

set(README_FILES image.txt)

find_package(PCL REQUIRED)
link_directories(${PCL_LIBRARY_DIRS})
include_directories(BEFORE ${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# find the DepthSense SDK
#list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
#find_package(DepthSenseSDK)

#if(NOT DepthSenseSDK_FOUND)
#  message(FATAL_ERROR "+++++++++ Could not locate the DepthSense SDK +++++++++
#Please install the SDK before trying to build the 'softkinetic_camera' package, see README.txt.

#Cannot continue, aborting.")
#  return()
#endif()

find_package(Boost COMPONENTS system filesystem thread REQUIRED)

find_package(PkgConfig)
pkg_check_modules(PC_LIBOPENNI REQUIRED libopenni)
pkg_check_modules(PC_LIBUSB REQUIRED libusb-1.0)
add_definitions(${PC_LIBOPENNI_CFLAGS_OTHER})

#generate_dynamic_reconfigure_options(cfg/OpenNI.cfg)


find_package(OpenCV REQUIRED)
find_package(CUDA)
IF(CUDA_FOUND)
  MESSAGE(STATUS "CUDA found")
  SET(HAVE_CUDA  FALSE)
  SET(HAVE_CUDA_FOUND "yes")
ELSE(CUDA_FOUND)
  MESSAGE(STATUS "CUDA not found")
  SET(HAVE_CUDA_FOUND "no") 
ENDIF(CUDA_FOUND)

FIND_PACKAGE(VISP)
#IF(VISP_FOUND)
#  INCLUDE(${VISP_USE_FILE})
#ENDIF(VISP_FOUND)
LINK_DIRECTORIES(${VISP_LINK_DIRECTORIES})


set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
# View the makefile commands during build
#set(CMAKE_VERBOSE_MAKEFILE on)


include( CheckCXXCompilerFlag )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

find_package(OpenGL)

set(DEPENDENCIES realsense2 /usr/lib/x86_64-linux-gnu/librealsense2.so ${OPENGL_LIBRARIES})

        # Find glfw header
        find_path(GLFW_INCLUDE_DIR NAMES GLFW/glfw3.h
            PATHS /usr/X11R6/include
                  /usr/include/X11
                  /opt/graphics/OpenGL/include
                  /opt/graphics/OpenGL/contrib/libglfw
                  /usr/local/include
                  /usr/include/GL
                  /usr/include
        )
        # Find glfw library
        find_library(GLFW_LIBRARIES NAMES glfw glfw3
                PATHS /usr/lib64
                      /usr/lib
                      /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                      /usr/local/lib64
                      /usr/local/lib
                      /usr/local/lib/${CMAKE_LIBRARY_ARCHITECTURE}
                      /usr/X11R6/lib
        )
        list(APPEND DEPENDENCIES m ${GLFW_LIBRARIES} ${LIBUSB1_LIBRARIES})
        include_directories(${GLFW_INCLUDE_DIR})

#include_directories(/home/antoine/soft/librealsense/examples /home/antoine/soft/librealsense/third-party/tclap/include)


FIND_PACKAGE(Lua51)
IF(NOT LUA51_FOUND)
    MESSAGE(CRITICAL Lua not found!)
ENDIF(NOT LUA51_FOUND)

find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)


# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
# View the makefile commands during build
#set(CMAKE_VERBOSE_MAKEFILE on)

set(${PROJECT_NAME}_INCLUDE_EXPORTS
  ${OpenCV_INCLUDE_DIRS}
  ${VISP_INCLUDE_DIR}
  ${VISP_INCLUDE_DIRS}
  ${LUA_INCLUDE_DIR}
  include
  /usr/local/cuda-6.5/include/
  /usr/local/cuda-6.5/samples/common/inc/
  /usr/local/cuda-6.5/samples/7_CUDALibraries/common/UtilNPP/
  #${DepthSenseSDK_INCLUDE_DIRS}
    /usr/include/pcl-1.8/
    ${EIGEN_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
	${PC_LIBOPENNI_INCLUDE_DIRS} 
	${Boost_INCLUDE_DIRS} 
/home/antoine/soft/librealsense/include
/home/antoine/soft/librealsense/include/librealsense2/
/opt/ros/indigo/include/
)

include_directories(
${PROJECT_NAME}_INCLUDE_EXPORTS
#${DepthSenseSDK_INCLUDE_DIRS}
)


set(ALL_LIBRARIES ${OpenCV_LIBS} freeimage ${LUA_LIBRARIES} ${FREEGLUT_LIBRARY} ${GLEW_LIBRARY} 
${OPENGL_LIBRARIES} ${CUDA_LIBRARY} ${CUDA_npp_LIBRARY} ${PCL_LIBRARIES} ${LIBS} 
#${DepthSenseSDK_LIBRARIES} 
${EIGEN_LIBRARIES}                                     
${Boost_LIBRARIES} ${PC_LIBOPENNI_LIBRARIES} ${PC_LIBUSB_LIBRARIES} /usr/lib/x86_64-linux-gnu/librealsense2.so)

if (NOT SOFA-MISC_NO_OPENGL)
    list(APPEND HEADER_FILES "ImageViewer.h")
    list(APPEND SOURCE_FILES "ImageViewer.cpp")
endif()

if(SOFA-EXTERNAL_LIBFREENECT)
    list(APPEND HEADER_FILES "Kinect.h")
    list(APPEND SOURCE_FILES "Kinect.cpp")
    AddLinkerDependencies("freenect")
    AddLinkerDependencies("freenect_sync")
endif()

if(SOFA-EXTERNAL_ZLIB)
    AddLinkerDependencies(${ZLIB_LIBRARIES})
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_IMAGE")
AddLinkerDependencies(SofaBaseVisual ${ALL_LIBRARIES})
# AddLinkerDependencies(${OPENGL_LIBRARIES})

if(SOFA-EXTERNAL_OPENCV)
    AddLinkerDependencies("ml")
    AddLinkerDependencies("cvaux")
    AddLinkerDependencies("highgui")
    AddLinkerDependencies("cv")
    AddLinkerDependencies("cxcore")
endif()

AddLinkerDependencies(newmat)

AddLinkerDependencies(${DEPENDENCIES})


include(${SOFA_CMAKE_DIR}/postProject.cmake)
