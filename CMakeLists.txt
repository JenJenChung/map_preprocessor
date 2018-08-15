cmake_minimum_required(VERSION 2.8.3)
project(map_preprocessor)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  gmapping
  map_server
  move_base
  roscpp
  rospy
  stage_ros
  message_generation
  std_msgs
  nav_msgs
  custom_messages
)

# Generate messages in the 'msg' folder
add_message_files(
  DIRECTORY
  FILES
  LocalMap.msg
  MapEncoding.msg
)

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  nav_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES map_preprocessor
#  CATKIN_DEPENDS gmapping map_server navigation roscpp rospy stage_ros
#  DEPENDS system_lib
)
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(planner_recover src/planner_recover.cpp)
target_link_libraries(planner_recover ${catkin_LIBRARIES})
add_dependencies(planner_recover ${catkin_EXPORTED_TARGETS})

add_executable(map_writer src/map_writer.cpp)
target_link_libraries(map_writer ${catkin_LIBRARIES})
add_dependencies(map_writer ${catkin_EXPORTED_TARGETS})

add_executable(local_map_state src/local_map_state.cpp)
target_link_libraries(local_map_state ${catkin_LIBRARIES})
add_dependencies(local_map_state ${catkin_EXPORTED_TARGETS})
