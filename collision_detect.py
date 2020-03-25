#!/usr/bin/env python

import rospy
import numpy as np
import tf
from math import sqrt, atan, atan2, pi
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

def callBack(msg):
    # print(msg.detections[0])
    pub_markers = rospy.Publisher('/vertex', MarkerArray, queue_size=100)

    # Delete all existing markers
    markers = MarkerArray()
    marker = Marker()
    marker.action = marker.DELETEALL
    markers.markers.append(marker)
    pub_markers.publish(markers)

    if len(msg.detections) > 1:
        C_a = np.array([msg.detections[0].bbox.center.position.x, msg.detections[0].bbox.center.position.y, msg.detections[0].bbox.center.position.z])
        C_b = np.array([msg.detections[1].bbox.center.position.x, msg.detections[1].bbox.center.position.y, msg.detections[1].bbox.center.position.z])
        V = C_b.transpose() - C_a.transpose() 
        print(V)
    
    tru_table = np.matrix([[-1,-1,-1,1], [-1,-1,1,1], [-1,1,-1,1], [-1,1,1,1], [1,-1,-1,1], [1,-1,1,1], [1,1,-1,1], [1,1,1,1]])
    cnt = 0

    for i in range(len(msg.detections)):
        p = np.array([msg.detections[i].bbox.size.x/2, msg.detections[i].bbox.size.y/2, msg.detections[i].bbox.size.z/2, 1])
        P = np.multiply(tru_table, p)

        trans = tf.TransformerROS(True,rospy.Duration(10.0))
        rotation = (msg.detections[i].bbox.center.orientation.x, msg.detections[i].bbox.center.orientation.y, msg.detections[i].bbox.center.orientation.z, msg.detections[i].bbox.center.orientation.w)
        translation = (msg.detections[i].bbox.center.position.x, msg.detections[i].bbox.center.position.y, msg.detections[i].bbox.center.position.z)
        T = trans.fromTranslationRotation(translation, rotation)

        P_t = np.matmul(T, P.transpose())

        for j in range(P_t.shape[1]):

            # cube marker
            marker = Marker()
            marker.header = msg.header
            marker.action = marker.ADD
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.position.x = P_t[:,j][0]
            marker.pose.position.y = P_t[:,j][1]
            marker.pose.position.z = P_t[:,j][2]
            marker.ns = "vertex"
            marker.id = cnt
            marker.type = marker.SPHERE
            markers.markers.append(marker)

            cnt += 1

    
    pub_markers.publish(markers)

def main():
    rospy.init_node('collision_detect')
    rospy.Subscriber("/detected_objects", Detection3DArray, callBack)
    rospy.spin()

if __name__ == '__main__':
    main()