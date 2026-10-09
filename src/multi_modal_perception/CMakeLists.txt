cmake_minimum_required(VERSION 3.10)
project(multi_modal_perception)

# 配置全局编译参数
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 核心依赖查找（ROS相关）
# 根据新代码的 #include，更新了组件列表
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  pcl_ros
  pcl_conversions
  cv_bridge
  visualization_msgs
)

# 系统级依赖配置
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if(NOT TARGET pcl_common)
find_package(PCL 1.10 REQUIRED COMPONENTS
  common
  io
  filters
  segmentation
  features
  search
  kdtree
)
endif()


find_package(OpenCV 4 REQUIRED COMPONENTS 
  core 
  imgproc 
  highgui
)

# 包级配置
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS 
    roscpp 
    sensor_msgs 
    std_msgs
    pcl_ros
    pcl_conversions
    cv_bridge
    visualization_msgs
  DEPENDS 
    Eigen3 
    PCL
    OpenCV
)

# 路径包含
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# 添加 PCL 定义
add_definitions(${PCL_DEFINITIONS})

# 可执行文件配置
add_executable(multi_modal_perception_node src/TargetDetect.cpp)

# 依赖链配置
add_dependencies(multi_modal_perception_node 
  ${catkin_EXPORTED_TARGETS}
)

# 链接器配置
target_link_libraries(multi_modal_perception_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

# 安装规则
install(TARGETS multi_modal_perception_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
