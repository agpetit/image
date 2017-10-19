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
	OpenniCam.cpp
	Openni2Cam.cpp

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
	OpenniCam.h
	Openni2Cam.h

)

set(README_FILES image.txt)

find_package(PCL REQUIRED)
link_directories(${PCL_LIBRARY_DIRS})
include_directories(BEFORE ${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# find the DepthSense SDK
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(DepthSenseSDK)

if(NOT DepthSenseSDK_FOUND)
  message(FATAL_ERROR "+++++++++ Could not locate the DepthSense SDK +++++++++
Please install the SDK before trying to build the 'softkinetic_camera' package, see README.txt.

Cannot continue, aborting.")
  return()
endif()

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
IF(VISP_FOUND)
  INCLUDE(${VISP_USE_FILE})
ENDIF(VISP_FOUND)
LINK_DIRECTORIES(${VISP_LINK_DIRECTORIES})

FIND_PACKAGE(Lua51)
IF(NOT LUA51_FOUND)
    MESSAGE(CRITICAL Lua not found!)
ENDIF(NOT LUA51_FOUND)

find_package(Eigen3 REQUIRED)

set(${PROJECT_NAME}_INCLUDE_EXPORTS
  ${OpenCV_INCLUDE_DIRS}
  ${VISP_INCLUDE_DIR}
  ${LUA_INCLUDE_DIR}
  include
  /usr/local/cuda-6.0/include/
  /usr/local/cuda-6.0/samples/common/inc/
  /usr/local/cuda-6.0/samples/7_CUDALibraries/common/UtilNPP/
  ${DepthSenseSDK_INCLUDE_DIRS}
  /usr/include/pcl-1.7/
    /usr/include/pcl-1.8/
    ${EIGEN_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
	${PC_LIBOPENNI_INCLUDE_DIRS} 
	${Boost_INCLUDE_DIRS} 
)

include_directories(
${PROJECT_NAME}_INCLUDE_EXPORTS
${DepthSenseSDK_INCLUDE_DIRS}

)


set(ALL_LIBRARIES ${OpenCV_LIBS} freeimage ${LUA_LIBRARIES} ${FREEGLUT_LIBRARY} ${GLEW_LIBRARY} 
${OPENGL_LIBRARIES} ${CUDA_LIBRARY} ${CUDA_npp_LIBRARY} ${PCL_LIBRARIES} 
${DepthSenseSDK_LIBRARIES} ${EIGEN_LIBRARIES}                                     
${Boost_LIBRARIES} ${PC_LIBOPENNI_LIBRARIES} ${PC_LIBUSB_LIBRARIES})

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


include(${SOFA_CMAKE_DIR}/postProject.cmake)
