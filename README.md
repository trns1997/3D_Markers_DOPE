# 3D_Markers_DOPE

```
$ roscore
$ rosbag play -l -r 0.04 takedown_depth.bag
$ rosrun tf2_ros static_transform_publisher 0 0 0 0.7071 0 0 -0.7071 world zed_left_camera_optical_frame
$ ./dope_rosbag.py
$ ./collision_detect.py
```
<img src="https://github.com/trns1997/" width="400" height="300"/>
