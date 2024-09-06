cmake_minimum_required(VERSION 3.0.2)
project(pretrained_models_client)

find_package(catkin_simple REQUIRED)
find_package(sensor_msgs REQUIRED)

catkin_simple(ALL_DEPS_REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)