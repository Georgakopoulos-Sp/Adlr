# Install script for directory: /home/legolas/Desktop/caffe_AdLR_EANN/python

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/legolas/Desktop/caffe_AdLR_EANN/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python" TYPE FILE FILES
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/mypython3.py"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/detect.py"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/draw_net.py"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/mypython2.py"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/mypython_features.py"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/mypython.py"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/classify.py"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/requirements.txt"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE FILE FILES
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/caffe/pycaffe.py"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/caffe/io.py"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/caffe/__init__.py"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/caffe/net_spec.py"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/caffe/detector.py"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/caffe/classifier.py"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/caffe/draw.py"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE SHARED_LIBRARY FILES "/home/legolas/Desktop/caffe_AdLR_EANN/python/CMakeFiles/CMakeRelink.dir/_caffe.so")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/python/caffe" TYPE DIRECTORY FILES
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/caffe/imagenet"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/caffe/proto"
    "/home/legolas/Desktop/caffe_AdLR_EANN/python/caffe/test"
    )
endif()

