#===================================================================#
# This program is the confidential and proprietary product of Valeo #
# Comfort and Driving Assistance Systems. Any unauthorised use,     #
# reproduction or transfer of this program is strictly prohibited.  #
#                                                                   #
# Copyright (C) 2018 Valeo Comfort and Driving Assistance Systems.  #
# (Subject to limited distribution and restricted disclosure only.) #
#===================================================================#

# Version Requirement
cmake_minimum_required(VERSION 2.8)

project(Face_detect)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set default CMAKE_BUILD_TYPE to Release
if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Find necessary packages
FIND_PACKAGE(PkgConfig REQUIRED)

#FIND_PACKAGE(Boost 1.63.0 REQUIRED COMPONENTS system filesystem thread)
message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost lib dirs: ${Boost_LIBRARY_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")

PKG_CHECK_MODULES(LIBSSL REQUIRED libssl)
message(STATUS "LIBSSL include dirs: ${LIBSSL_INCLUDE_DIRS}")
message(STATUS "LIBSSL lib dirs: ${LIBSSL_LIBRARY_DIRS}")

PKG_CHECK_MODULES(LIBCRYPTO REQUIRED libcrypto)
message(STATUS "LIBCRYPTO include dirs: ${LIBCRYPTO_INCLUDE_DIRS}")
message(STATUS "LIBCRYPTO lib dirs: ${LIBCRYPTO_LIBRARY_DIRS}")

#find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui models)
FIND_PACKAGE(OpenCV REQUIRED)

set(App_TARGET "${PROJECT_NAME}")

add_executable(${App_TARGET}
  src/main.cpp
  src/image.cpp
  src/algorithm.cpp
)

set(APP_TARGET ${PROJECT_NAME})

target_include_directories (${APP_TARGET} PRIVATE
  ${OpenCV_INCLUDE_DIRS}
#  ${Boost_INCLUDE_DIRS}
#  ${LIBSSL_INCLUDE_DIRS}
#  ${LIBCRYPTO_INCLUDE_DIRS}
 # .
)

target_link_libraries (${APP_TARGET}
  ${OpenCV_LIBS}
 # ${Boost_LIBRARIES}
  #${LIBSSL_LIBRARIES}
 # ${LIBCRYPTO_LIBRARIES}
)

install(
  TARGETS ${App_TARGET}
  RUNTIME DESTINATION bin
  COMPONENT ${PROJECT_NAME}
)
