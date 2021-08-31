#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import math
from cv_bridge import CvBridge, CvBridgeError
import cv2
import message_filters
from std_msgs.msg import Float64MultiArray 
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs import point_cloud2


pre_position = np.zeros((1 , 4))
position_experience = np.zeros((1 , 4))
image_save_path = "/home/zach/catkin_ws/save_image/"
event_list = np.zeros(15)
time_decline = 0.05


EM_weight = np.ones((1 , 15))
EM_node_num = 1
EM_threshold = 0.8

Episodic_memory = np.zeros((1 , 15))
EM_MAP = np.empty((1 , 2) , dtype = int)
path = np.zeros((1 , 4))

class CompareImage(object):


    def __init__(self, image_1_path, image_2_path):
        self.minimum_commutative_image_diff = 1
        self.image_1_path = image_1_path
        self.image_2_path = image_2_path
        #print(image_1_path)
        #print(image_2_path)
    def compare_image(self):
        image_1 = cv2.imread(self.image_1_path)
        image_2 = cv2.imread(self.image_2_path)
        commutative_image_diff = self.get_image_difference(image_1 , image_2)

        if commutative_image_diff < self.minimum_commutative_image_diff:
            print("Matched")
            return commutative_image_diff
        return 10000
    
    def get_image_difference(self , image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        #print(img_hist_diff)
        img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match
        #print(img_template_diff)
        # taking only 10% of histogram diff, since it's less accurate than template method
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        #print(commutative_image_diff)
        return commutative_image_diff






def string_to_float(str):
    return float(str)

def martix_move(martix , shift1 , shift2):
    h , w = matrix.shape
    matrix=np.vstack((matrix[(h-shiftnum1):,:],matrix[:(h-shiftnum1),:]))
    matrix=np.hstack((matrix[:,(w-shiftnum2):],matrix[:,:(w-shiftnum2)]))
    return matrix



def EMcallback(data1 , data2 , data3):
    cv_img = bridge.imgmsg_to_cv2(data2, "bgr8")
    episodic_flag = 0
    #publish conmand string init
    #em_pub = rospy.Publisher("conmand" , String  ,queue_size=10)
    # data1 = position ; data2 = image ; data3 = depth_point 
    
    #position fuzzy
    '''
    now the program could remember the path
    the path includes the position x , y, and the head
    '''
    '''x = data1.data[0]
    y = data1.data[1]     it is not good to get the postion from my node, now I get the position infromation
    oz = data1.data[5]    from the odom
    ow = data1.data[6]'''
    x = data1.pose.pose.position.x
    y = data1.pose.pose.position.y
    oz = data1.pose.pose.orientation.z
    ow = data1.pose.pose.orientation.w
    #theta = data1.pose.pose.theta
    #print(x , y , oz , ow)
    x = ('%.1f' %x)
    y = ('%.1f' %y)
    oz = ('%.1f' %oz)
    ow = ('%.1f' %ow)
    x = string_to_float(x)
    y = string_to_float(y)
    oz = string_to_float(oz)
    ow = string_to_float(ow)
    
    plan_node = []    
    now_position = np.array([x , y , oz , ow])
    global path
    path = np.vstack([path , now_position])
    np.savetxt('/home/zach/catkin_ws/path' , path)
    #print(now_position)
    position_node = False
    global position_experience
    experience_num = np.shape(position_experience)[0]
    #print(experience_num)
    save_list = np.array([])

    temp_img = "/home/zach/catkin_ws/save_image/temp_image.jpg"
    cv2.imwrite(temp_img , cv_img)
    
    for i in range(experience_num):
        if(now_position == position_experience[i]).all():
            position_node = i
            i = str(i)
            called_image = image_save_path + i + '.jpg'
            print(position_node)
            compare_image = CompareImage(called_image, temp_img)
            image_difference = compare_image.compare_image()
            if(image_difference == 10000):
                print("update image")



            
    if(position_node == False):
        for i in range(experience_num):
            i = str(i)
            called_image = image_save_path + i + '.jpg'
            compare_image = CompareImage(called_image , temp_img)
            image_difference = compare_image.compare_image()
            if(image_difference != 10000):
                print("same event")
                position_node = i
                break
            else:
                position_experience = np.row_stack((position_experience , now_position))
                experience_num = experience_num + 1
                experience_num_1 = str(experience_num)
                image_name = experience_num_1 + '.jpg'
                cv2.imwrite(image_save_path + image_name , cv_img)
                position_node = experience_num
                np.savetxt('/home/zach/catkin_ws/map' , position_experience)
                #np.savetxt('/home/zach/catkin_ws/weight' , )
        
       
    #print(position_experience)
    print(position_node)
    global event_list
    #print("ep" , event_list)
    event_list_temp = event_list[1:]
    #print("et" , event_list_temp)
    event_list[:-1] = event_list_temp
    #print("e" , event_list)
    event_list[14] = position_node
    fuzzy_num = 0
    weight_and = 0
    global EM_node_num
    activate_num = np.zeros(EM_node_num)
    global EM_weight
    #print(event_list)
    #print(EM_weight)

    for i in range(0 , EM_node_num):
        for j in range(15):
            min_num = min(event_list[j] , EM_weight[i , j])
            min_num = abs(min_num)
            fuzzy_num = fuzzy_num + min_num
            weight_and = weight_and + abs(EM_weight[i , j])
        activate_num[i] = fuzzy_num / (1 + weight_and)
    activate_node = np.argmax(activate_num)
    weight_activate = EM_weight[activate_node]

    simlation = fuzzy_num / weight_and
    #activate
    if(simlation > EM_threshold):
        EM_weight[activate_node] = 0.7 * EM_weight[activate_node] + 0.3 * event_list
        print("now event list" , activate_node)
        '''
            while(activate_node < EM_node_num):
            plan_node = Episodic_memory[activate_node]
            activate_node = activate_node + 1
        '''
    #study
    else: 
        EM_weight = np.vstack((EM_weight , event_list))
        EM_node_num = EM_node_num + 1
        global Episodic_memory
        #Episodic_memory[EM_node_num] = event_list
        #Episodic_memory = np.append((Episodic_memory , event_list))
        #Episodic_memory.append(event_list)
        #Episodic_memory = np.array(Episodic_memory)
        Episodic_memory = np.vstack([Episodic_memory , event_list])
        print("new event list" , EM_node_num)
        episodic_flag = 1

    np.savetxt('/home/zach/catkin_ws/memory' , Episodic_memory , fmt='%.0f') #
    #EM_length = np.shape(Episodic_memory)[0]
    #print(Episodic_memory)


    if(episodic_flag == 1):
        temp = 0
        event_length = np.shape(Episodic_memory)[0]
        #print("shijianshu" , event_length)
        em_length = np.shape(Episodic_memory)[1]
        #print("shijianchangdu" , em_length)
        if(em_length > 15):
            em_length = 15
        for i_1 in range(event_length):
            for j_1 in range(em_length):
                temp = Episodic_memory[i_1 , j_1]
                temp = int(temp)
                print(temp)
                global EM_MAP
                EM_MAP = np.vstack((EM_MAP , position_experience[temp-1 , :2]))
    np.savetxt('/home/zach/catkin_ws/EMmap' , EM_MAP)



def listener():
    rospy.init_node('EEMART' , anonymous=True)
    global bridge
    bridge = CvBridge()
    #position_event = rospy.Subscriber("location_event" , String)
    image = message_filters.Subscriber('/camera/rgb/image_raw', Image)
    depth_point = message_filters.Subscriber('/camera/depth/points' , PointCloud2)
    position_image = message_filters.Subscriber('/odom' , Odometry )
    image_pos = message_filters.ApproximateTimeSynchronizer([position_image , image , depth_point] , 10 , 1 ,allow_headerless = True)
    image_pos.registerCallback(EMcallback)
    rospy.spin()


if __name__ == "__main__":
    listener()


