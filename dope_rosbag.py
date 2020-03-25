#!/usr/bin/env python

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

"""
This file starts a ROS node to run DOPE, 
listening to an image topic and publishing poses.
"""

from __future__ import print_function
import yaml
import sys 

import numpy as np
import cv2

import rospy
import rospkg
import tf.transformations
from std_msgs.msg import String, Empty
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ImageSensor_msg
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from PIL import Image
from PIL import ImageDraw
# load and display an image with Matplotlib
#from PIL import Pyplot

# Import DOPE code
rospack = rospkg.RosPack()
g_path2package = rospack.get_path('dope')
sys.path.append("{}/src/inference".format(g_path2package))
from dope.inference.cuboid import Cuboid3d
from dope.inference.cuboid_pnp_solver import CuboidPNPSolver
from dope.inference.detector import ModelData, ObjectDetector

### Global Variables
g_bridge = CvBridge()
g_img = None
g_draw = None
g_img_count = 0
image_msg = None

### Basic functions
def __image_callback(msg):
    '''Image callback'''
    global g_img
    global g_img_count
    global image_msg
    image_msg = msg
    g_img = g_bridge.imgmsg_to_cv2(msg, "rgb8")
    # Downscale image if necessary
    height, width, _ = g_img.shape
    scaling_factor = float(400) / height
    if scaling_factor < 1.0:
        g_img = cv2.resize(g_img, (int(scaling_factor * width), int(scaling_factor * height)))
    name = "img" + str(g_img_count) + ".png"
    g_img_count += 1
    #print("Image:" + str(g_img_count))
    #cv2.imwrite(name, cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB))  # for debugging

    #g_img = Image.open(g_img)
    # summarize some details about the image
    #print("Mode:" + str(g_img.shape))


### Code to visualize the neural network output

def DrawLine(point1, point2, lineColor, lineWidth):
    '''Draws line on image'''
    global g_draw
    if not point1 is None and point2 is not None:
        g_draw.line([point1,point2], fill=lineColor, width=lineWidth)

def DrawDot(point, pointColor, pointRadius):
    '''Draws dot (filled circle) on image'''
    global g_draw
    if point is not None:
        xy = [
            point[0]-pointRadius, 
            point[1]-pointRadius, 
            point[0]+pointRadius, 
            point[1]+pointRadius
        ]
        g_draw.ellipse(xy, 
            fill=pointColor, 
            outline=pointColor
        )

def DrawCube(points, color=(255, 0, 0)):
    '''
    Draws cube with a thick solid line across 
    the front top edge and an X on the top face.
    '''

    lineWidthForDrawing = 2

    # draw front
    DrawLine(points[0], points[1], color, lineWidthForDrawing)
    DrawLine(points[1], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[2], color, lineWidthForDrawing)
    DrawLine(points[3], points[0], color, lineWidthForDrawing)
    
    # draw back
    DrawLine(points[4], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[5], color, lineWidthForDrawing)
    DrawLine(points[6], points[7], color, lineWidthForDrawing)
    DrawLine(points[4], points[7], color, lineWidthForDrawing)
    
    # draw sides
    DrawLine(points[0], points[4], color, lineWidthForDrawing)
    DrawLine(points[7], points[3], color, lineWidthForDrawing)
    DrawLine(points[5], points[1], color, lineWidthForDrawing)
    DrawLine(points[2], points[6], color, lineWidthForDrawing)

    # draw dots
    DrawDot(points[0], pointColor=color, pointRadius = 4)
    DrawDot(points[1], pointColor=color, pointRadius = 4)

    # draw x on the top 
    DrawLine(points[0], points[5], color, lineWidthForDrawing)
    DrawLine(points[1], points[4], color, lineWidthForDrawing)


def run_dope_node(params, freq=5):
    '''Starts ROS node to listen to image topic, run DOPE, and publish DOPE results'''

    global g_img
    global g_draw

    pubs = {}
    models = {}
    pnp_solvers = {}
    pub_dimension = {}
    draw_colors = {}
    class_ids = {}
    model_transforms = {}
    dimensions = {}
    mesh_scales = {}
    meshes = {}

    # Initialize parameters
    matrix_camera = np.zeros((3,3))
    matrix_camera[0,0] = params["camera_settings"]['fx']
    matrix_camera[1,1] = params["camera_settings"]['fy']
    matrix_camera[0,2] = params["camera_settings"]['cx']
    matrix_camera[1,2] = params["camera_settings"]['cy']
    matrix_camera[2,2] = 1
    dist_coeffs = np.zeros((4,1))

    if "dist_coeffs" in params["camera_settings"]:
        dist_coeffs = np.array(params["camera_settings"]['dist_coeffs'])
    config_detect = lambda: None
    config_detect.mask_edges = 1
    config_detect.mask_faces = 1
    config_detect.vertex = 1
    config_detect.threshold = 0.5
    config_detect.softmax = 1000
    config_detect.thresh_angle = params['thresh_angle']
    config_detect.thresh_map = params['thresh_map']
    config_detect.sigma = params['sigma']
    config_detect.thresh_points = params["thresh_points"]

    # For each object to detect, load network model, create PNP solver, and start ROS publishers
    for model in params['weights']:
        models[model] =\
            ModelData(
                model, 
                g_path2package + "/weights/" + params['weights'][model]
            )
        models[model].load_net_model()
        mesh_scales[model] = 1.0
        model_transforms[model] = np.array([0.0, 0.0, 0.0, 1.0], dtype='float64')
        draw_colors[model] = \
            tuple(params["draw_colors"][model])
        class_ids[model] = \
            (params["class_ids"][model])
        dimensions[model] = tuple(params["dimensions"][model])
        pnp_solvers[model] = \
            CuboidPNPSolver(
                model,
                matrix_camera,
                Cuboid3d(params['dimensions'][model]),
                dist_coeffs=dist_coeffs
            )
        pubs[model] = \
            rospy.Publisher(
                '{}/pose_{}'.format(params['topic_publishing'], model), 
                PoseStamped, 
                queue_size=10
            )
        pub_dimension[model] = \
            rospy.Publisher(
                '{}/dimension_{}'.format(params['topic_publishing'], model),
                String, 
                queue_size=10
            )

    # Start ROS publisher
    pub_rgb_dope_points = \
        rospy.Publisher(
            params['topic_publishing']+"/rgb_points", 
            ImageSensor_msg, 
            queue_size=10
        )
    pub_detections = \
        rospy.Publisher(
            'detected_objects',
            Detection3DArray,
            queue_size=10
        )
    pub_markers = \
            rospy.Publisher(
                'markers',
                MarkerArray,
                queue_size=10
            )
    
    # Starts ROS listener
    rospy.Subscriber(
        topic_cam, 
        ImageSensor_msg, 
        __image_callback
    )

    # Initialize ROS node
    rospy.init_node('dope_vis', anonymous=True)
    rate = rospy.Rate(freq)

    print ("Running DOPE...  (Listening to camera topic: '{}')".format(topic_cam)) 
    print ("Ctrl-C to stop")

    while not rospy.is_shutdown():
        if g_img is not None:
            # Copy and draw image
            img_copy = g_img.copy()
            im = Image.fromarray(img_copy)
            g_draw = ImageDraw.Draw(im)

            detection_array = Detection3DArray()
            detection_array.header = image_msg.header


            for m in models:
                # Detect object
                results = ObjectDetector.detect_object_in_image(
                            models[m].net, 
                            pnp_solvers[m],
                            g_img,
                            config_detect
                            )
                
                # Publish pose and overlay cube on image
                for i_r, result in enumerate(results):
                    if result["location"] is None:
                        continue
                    loc = result["location"]
                    ori = result["quaternion"]  

                    # transform orientation
                    transformed_ori = tf.transformations.quaternion_multiply(ori, model_transforms[m])

                    # rotate bbox dimensions if necessary
                    # (this only works properly if model_transform is in 90 degree angles)
                    dims = rotate_vector(vector=dimensions[m], quaternion=model_transforms[m])
                    dims = np.absolute(dims)
                    dims = tuple(dims)
                                    
                    msg = PoseStamped()
                    msg.header.frame_id = params["frame_id"]
                    msg.header.stamp = rospy.Time.now()
                    CONVERT_SCALE_CM_TO_METERS = 100
                    msg.pose.position.x = loc[0] / CONVERT_SCALE_CM_TO_METERS
                    msg.pose.position.y = loc[1] / CONVERT_SCALE_CM_TO_METERS
                    msg.pose.position.z = loc[2] / CONVERT_SCALE_CM_TO_METERS
                    msg.pose.orientation.x = ori[0]
                    msg.pose.orientation.y = ori[1]
                    msg.pose.orientation.z = ori[2]
                    msg.pose.orientation.w = ori[3]

                    # Publish
                    pubs[m].publish(msg)
                    pub_dimension[m].publish(str(params['dimensions'][m]))

                    # Add to Detection3DArray
                    detection = Detection3D()
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.id = class_ids[result["name"]]
                    hypothesis.score = result["score"]
                    hypothesis.pose.pose = msg.pose
                    detection.results.append(hypothesis)
                    detection.bbox.center = msg.pose
                    detection.bbox.size.x = dims[0] / CONVERT_SCALE_CM_TO_METERS
                    detection.bbox.size.y = dims[1] / CONVERT_SCALE_CM_TO_METERS
                    detection.bbox.size.z = dims[2] / CONVERT_SCALE_CM_TO_METERS
                    detection_array.detections.append(detection)

                    # Draw the cube
                    if None not in result['projected_points']:
                        points2d = []
                        for pair in result['projected_points']:
                            points2d.append(tuple(pair))
                        DrawCube(points2d, draw_colors[m])
                
            # Publish the image with results overlaid
            pub_rgb_dope_points.publish(
                CvBridge().cv2_to_imgmsg(
                    np.array(im)[...,::-1], 
                    "bgr8"
                )
            )

            # Delete all existing markers
            markers = MarkerArray()
            marker = Marker()
            marker.action = Marker.DELETEALL
            markers.markers.append(marker)
            pub_markers.publish(markers)

            # Object markers
            class_id_to_name = {class_id: name for name, class_id in class_ids.iteritems()}
            markers = MarkerArray()
            for i, det in enumerate(detection_array.detections):
                name = class_id_to_name[det.results[0].id]
                color = draw_colors[name]

                # cube marker
                marker = Marker()
                marker.header = detection_array.header
                marker.action = Marker.ADD
                marker.pose = det.bbox.center
                marker.color.r = color[0] / 255.0
                marker.color.g = color[1] / 255.0
                marker.color.b = color[2] / 255.0
                marker.color.a = 0.4
                marker.ns = "bboxes"
                marker.id = i
                marker.type = Marker.CUBE
                marker.scale = det.bbox.size
                markers.markers.append(marker)
            
            pub_markers.publish(markers)
            pub_detections.publish(detection_array)

        rate.sleep()

def rotate_vector(vector, quaternion):
    q_conj = tf.transformations.quaternion_conjugate(quaternion)
    vector = np.array(vector, dtype='float64')
    vector = np.append(vector, [0.0])
    vector = tf.transformations.quaternion_multiply(q_conj, vector)
    vector = tf.transformations.quaternion_multiply(vector, quaternion)
    return vector[:3]


if __name__ == "__main__":
    '''Main routine to run DOPE'''

    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        config_name = "config_pose_rosbag.yaml"
    rospack = rospkg.RosPack()
    params = None
    yaml_path = g_path2package + '/config/{}'.format(config_name)
    with open(yaml_path, 'r') as stream:
        try:
            print("Loading DOPE parameters from '{}'...".format(yaml_path))
            params = yaml.load(stream)
            print('    Parameters loaded.')
        except yaml.YAMLError as exc:
            print(exc)

    topic_cam = params['topic_camera']

    try :
        run_dope_node(params)
    except rospy.ROSInterruptException:
        pass

