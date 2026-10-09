# TargetDetect

TargetDetect is an RGB–LiDAR obstacle point-cloud extraction system implemented with ROS and Python. The ROS node processes synchronized RGB images and LiDAR point clouds, while the Python server performs YOLO-based instance segmentation.

## Requirements

- Ubuntu
- ROS 1
- Catkin
- CMake
- C++ compiler with C++14 support
- Python 3
- OpenCV
- Point Cloud Library
- Python dependencies required by `yolo_segmentation_inference.py`

Ensure that ROS is installed and properly configured before building the project.

## Workspace Structure

Create a Catkin workspace and place the provided `src` directory inside it. The resulting directory structure should be:

```text
catkin_ws/
└── src/
    ├── multi_modal_perception/
    └── PythonServer/
```

## Build the ROS Workspace

Open a terminal and create the Catkin workspace:

```bash
mkdir -p catkin_ws
```

Place the provided `src` directory inside `catkin_ws`, then enter the workspace:

```bash
cd catkin_ws
```

Build the project:

```bash
catkin_make
```

The build must complete successfully before running the Python server or ROS node.

## Run the YOLO Instance Segmentation Server

In the first terminal, enter the Python server directory:

```bash
cd src/PythonServer
```

Start the YOLO instance segmentation server:

```bash
python3 yolo_segmentation_inference.py
```

Keep this terminal open while the system is running. The Python server must remain active so that the ROS node can request instance-segmentation results.

## Run the ROS Node

Open a second terminal and enter the Catkin workspace:

```bash
cd catkin_ws
```

If the workspace has not yet been built in the current environment, run:

```bash
catkin_make
```

Source the workspace environment:

```bash
source devel/setup.bash
```

Run the multi-modal perception node:

```bash
rosrun multi_modal_perception multi_modal_perception_node
```

## Recommended Startup Order

1. Build the Catkin workspace.
2. Start the YOLO instance segmentation server.
3. Open a second terminal.
4. Source the Catkin workspace environment.
5. Ensure that the required ROS master and sensor topics are available.
6. Start the multi-modal perception node.

## Notes

- The Python segmentation server must be started before the ROS perception node.
- Run `source devel/setup.bash` in every new terminal used to execute the ROS package.
- The RGB image and LiDAR point-cloud topics must be available before the perception node can produce results.
- If a ROS bag is used as the data source, start bag playback after the Python server and ROS node are ready.
- Model weights and configuration paths should be checked before running the Python server.
