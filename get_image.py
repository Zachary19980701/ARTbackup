#!/usr/bin/env python


import rospy
import numpy as np
from sensor_msgs.msg import Image
#from sensor_msgs.msg import PointCloud2
import math
from cv_bridge import CvBridge, CvBridgeError
import cv2
import message_filters
from std_msgs.msg import Float64MultiArray 
from std_msgs.msg import String
from nav_msgs.msg import Odometry

image_save_path = "/home/zac/catkin_ws/train_image/"
path = np.zeros((1 , 4))
pre_position = np.zeros((1 , 4))
position_experience = np.zeros((1 , 4))
experience_num = np.shape(position_experience)[0]


def string_to_float(str):
    return float(str)



def image_callback(data1 , data2):

    save_flag = 1
    cv_img = bridge.imgmsg_to_cv2(data2, "bgr8")
    x = data1.pose.pose.position.x
    y = data1.pose.pose.position.y
    oz = data1.pose.pose.orientation.z
    ow = data1.pose.pose.orientation.w

    x = ('%.1f' %x)
    y = ('%.1f' %y)
    oz = ('%.1f' %oz)
    ow = ('%.1f' %ow)
    x = string_to_float(x)
    y = string_to_float(y)
    oz = string_to_float(oz)
    ow = string_to_float(ow)
    now_position = np.array([x , y , oz , ow])
    global path
    path = np.vstack([path , now_position])
    np.savetxt('/home/zac/catkin_ws/path' , path)
    global position_experience
    experience_num = np.shape(position_experience)[0]
    print("now" , now_position)
    #print(position_experience)
    for i in range(experience_num):
        if(now_position == position_experience[i]).all():
            save_flag = 0
            break
        

    if(save_flag == 1):
        print(i , position_experience[i])
        position_experience = np.row_stack((position_experience , now_position))
        experience_num = experience_num + 1
        experience_num_1 = str(experience_num)
        image_name = experience_num_1 + '.jpg'
        cv2.imwrite(image_save_path + image_name , cv_img)
        print("save image")







def listener():
    rospy.init_node('get_image' , anonymous=True)
    global bridge
    bridge = CvBridge()
    #position_event = rospy.Subscriber("location_event" , String)
    image = message_filters.Subscriber('/camera/rgb/image_raw', Image)
    position_image = message_filters.Subscriber('/odom' , Odometry )
    image_pos = message_filters.ApproximateTimeSynchronizer([position_image , image ] , 10 , 1 ,allow_headerless = True)
    image_pos.registerCallback(image_callback)
    rospy.spin()
 

if __name__ == "__main__":
    listener()