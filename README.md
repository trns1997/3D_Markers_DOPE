# 3D_Markers_DOPE

```
$ roscore
$ rosbag play -l -r 0.04 takedown_depth.bag
$ rosrun tf2_ros static_transform_publisher 0 0 0 0.7071 0 0 -0.7071 world zed_left_camera_optical_frame
$ ./dope_rosbag.py
$ ./collision_detect.py
```
<img src="https://github.com/trns1997/3D_Markers_DOPE/blob/master/media/res.png" width="500" height="300"/>

#### Reference Respository: https://github.com/NVlabs/Deep_Object_Pose
