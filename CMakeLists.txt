#############################################
# TODDetector project
#############################################
cmake_minimum_required(VERSION 2.8)

project(TODDetector)
set(TODDetector_VERSION_MAJOR 1)
set(TODDetector_VERSION_MINOR 0)

#############################################
# Dependencies

#OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(SEND_ERROR "Faild to find Opencv")
    return()
else()
    set(TOD_DEPENDENCY_DIRS ${OpenCV_INCLUDE_DIRS} ${TOD_DEPENDENCY_DIRS})
    set(TOD_DEPENDENCY_LIBS ${OpenCV_LIBS} ${TOD_DEPENDENCY_LIBS})
endif()

find_package(PCL REQUIRED)
    set(TOD_DEPENDENCY_DIRS ${PCL_INCLUDE_DIRS} ${TOD_DEPENDENCY_DIRS})
    set(TOD_DEPENDENCY_LIBS ${PCL_LIBRARIES} ${TOD_DEPENDENCY_LIBS})
#############################################


#############################################
#General compiler flags

add_definitions(-Wall) #Enable all warnings always
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x -mtune=corei7 -march=corei7")
#############################################

include_directories(${TOD_DEPENDENCY_DIRS})

##################################
# Sources

file(GLOB main_cpp ./main.cpp)
file(GLOB_RECURSE sources_cpp src/*.cpp)
file(GLOB_RECURSE sources_h include/*.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
##################################

##################################
# Target

add_library(toddetector ${main_cpp} ${sources_cpp} ${sources_h})
target_link_libraries(toddetector ${TOD_DEPENDENCY_LIBS})
##################################


