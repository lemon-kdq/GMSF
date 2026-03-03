cmake_minimum_required(VERSION 3.0.2)
project(gt_pose_batch_optimization)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -W -Wno-unused-parameter}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin_simple 0.1.0 REQUIRED )
catkin_simple(ALL_DEPS_REQUIRED) 
catkin_python_setup()


cs_install()
