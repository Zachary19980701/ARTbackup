import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.vq import *
from sklearn import preprocessing
from PIL import Image
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import pickle

error_num = 0
loop_num = 0

activate_episodic = np.zeros((1 , 2))

loop = []
episodic_num = []
event_num = []
episodic_map = np.zeros((1 , 2))
#读取词典信息，BOW部分
voc = open(r"D:/project/corridor_data/voc.pkl" , "rb")
voc = pickle.load(voc)
idf = open(r"D:/project/corridor_data/idf.pkl" , "rb")
idf = pickle.load(idf)
database = np.ones((1 , 32))
numwords = 32
sift = cv2.xfeatures2d.SIFT_create()


EM_weight = np.zeros((1 , 72))
EEM_weight = np.zeros((1 , 1000))


event_list = np.zeros((10 , 100))

#读取位姿信息
gesture = np.loadtxt("D:/project/corridor_data/map1")

x = gesture[: , 0] + 6
y = gesture[: , 1] + 6
o_z = (gesture[: , 2] + 1) / 2
o_w = (gesture[: , 3] + 1) / 2

#map = np.loadtxt("D:/project/corridor_data/map1")
'''
real_positon = np.loadtxt("D:/project/corridor_data/map1")
rx = real_positon[: , 0] + 6
ry = real_positon[: , 1] + 6
ro_z = (real_positon[: , 2] + 1) / 2
ro_w = (real_positon[: , 3] + 1) / 2

rx = rx / 13
ry = ry / 8
'''


for i in range(len(x)):
    print("restart" , i)
    event = 0
    # 输入数据预处理
    input_x = x[i] / 13
    input_y = y[i] / 8
    input_oz = o_z[i]
    input_ow = o_w[i]
    input_img = i + 1
    input_img = str(input_img)
    img_name = input_img + ").jpg"
    img_path = "D:/project/corridor_data/corridor_image1/" + "p (" + img_name
    img = cv2.imread(img_path)
    # 图像数据转化为词袋模型

    des_list = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    des_list.append((img_path, des))
    descriptors = des_list[0][1]

    test_features = np.zeros((1, numwords), "float32")
    words, distance = vq(descriptors, voc)
    for w in words:
        test_features[0][w] += 1
    test_features = test_features * idf
    test_features = preprocessing.normalize(test_features, norm='l2')
    # print(test_features)
    # 输入向量归一化
    input_data = np.zeros(72)

    input_data[0] = input_x
    input_data[1] = input_y
    input_data[2] = input_oz
    input_data[3] = input_ow
    input_data[4: 36] = -test_features[0: 32]
    input_data_1 = 1 - input_data

    input_data[36: 72] = input_data_1[0: 36]
    # print("input data" , input_data)
    # 事件节点ART判断
    node_num = np.zeros(EM_weight.shape[0])
    # print(EM_weight.shape[0])
    for j in range(EM_weight.shape[0]):
        min_and = 0
        for k in range(72):
            # print(EM_weight[j][k])
            # print(input_data[k])
            min_num = min(input_data[k], EM_weight[j][k])
            min_and = min_and + min_num
        # print(min_and)
        em_and = sum(EM_weight[j])
        # print(em_and)
        node_num[j] = 0.5 * min_and / (0.1 + em_and)
    # print(node_num)
    activate_node = np.argsort(-node_num)
    activate_node = activate_node[0]
    # print(activate_node)

    # 相似度计算
    min_and = 0
    for j in range(72):
        min_num = min(input_data[j], EM_weight[activate_node][j])
        min_and = min_and + min_num
    # print("min_and" , min_and)
    em_and = sum(input_data)
    simlation = min_and / em_and
    # print(min_and)
    print("em_and", em_and)
    print("sim", simlation)
    if (simlation < 0.95):
        EM_weight = np.vstack((EM_weight, input_data))
        print("new node", EM_weight.shape[0])
        event = EM_weight.shape[0]
        database = np.vstack((database, test_features))
    else:
        EM_weight[activate_node] = 0.05 * EM_weight[activate_node] + 0.95 * input_data
        print("update node", activate_node)
        event = activate_node
    '''
    scores = np.dot(test_features, database.T)

    if (scores[0][0] < 0.8):
        database = np.vstack((database, test_features))
    '''
    em_node = np.zeros(100)
    em_node[event] = 1

    event_list_temp = event_list[1:10 , :]
    event_list[0:9 , :] = event_list_temp
    event_list[9 , :] = em_node
    # 衰减计算

    event_list[0] = event_list[0] * 0.1
    event_list[1] = event_list[1] * 0.2
    event_list[2] = event_list[2] * 0.3
    event_list[3] = event_list[3] * 0.4
    event_list[4] = event_list[4] * 0.5
    event_list[5] = event_list[5] * 0.6
    event_list[6] = event_list[6] * 0.7
    event_list[7] = event_list[7] * 0.8
    event_list[8] = event_list[8] * 0.9
    event_list[9] = event_list[9] * 1.0
    

    input_list = event_list.reshape(1000)
    # 计算情景地图
    node_num = np.zeros(EEM_weight.shape[0])
    em_activate_num = np.zeros(EEM_weight.shape[0])
    for j in range(EEM_weight.shape[0]):
        # 计算signification
        signification = 0
        min_and = 0
        sum_weight = 0
        for k in range(1000):
            em_min = min(input_list[k], EEM_weight[j][k])
            min_and = min_and + em_min
        sum_weight = sum(EEM_weight[j])
        if(sum_weight == 0):
            sum_weight = 1000
        signification = min_and / sum_weight

        # 计算similarity
        # 模值计算
        max_and = 0

        sim_up1 = np.dot(input_list, EEM_weight[j].T)
        sim_down1 = (np.linalg.norm(input_list) * np.linalg.norm(input_list)) * (
                    np.linalg.norm(EEM_weight[j]) * np.linalg.norm(EEM_weight[j]))
        if(sim_down1 == 0):
            sim_down1  = 1000
        sim1 = sim_up1 / sim_down1
        #print("sim1" , sim1)

        for k in range(1000):
            em_max = max(input_list[k], EEM_weight[j][k])
            max_and = max_and + em_max
        sim2 = min_and / max_and
        #print("sim2" , sim2)
        simliarity = (sim1 + sim2) / 2
        em_activate_num[j] = signification * (1 - (1 - simliarity))
        # print("em_activate_num" , em_activate_num)
    em_activate_node = np.argsort(-em_activate_num)[0]
    '''
    threshold_and = 0
    for k in range(1000):
        min_and_1 = min(input_list[k], EEM_weight[em_activate_node][k])
        threshold_and = threshold_and + min_and_1
    b = sum(input_list)
    threshold = min_and_1 / b
    print("threshold" , threshold)
    if(threshold < 0.99):
        EEM_weight = np.vstack((EEM_weight, input_list))
        print("new episodic", EEM_weight.shape[0])
    else:
        print("update episodic", em_activate_node)
    '''
    if(EEM_weight[em_activate_node] == input_list).all():
        loop_num += 1
        print("the line is" , em_activate_node)
        #激活先前情景，进行回环检测。
        memory = EEM_weight[em_activate_node].reshape(10 , 100)
        predict_event = 0
        activate_episodic[0 , 0] = input_x
        activate_episodic[0 , 1] = input_y
        for l in range(10):
            predict_event = np.argsort(-memory[l])[0]
            print(predict_event)
            scores = np.dot(test_features, database[predict_event].T)
            if(scores > 0.9):

                print("loop closing" , scores)
                ux = abs(input_data[0] - EM_weight[predict_event][0])
                uy = abs(input_data[1] - EM_weight[predict_event][1])
                uroz = abs(input_data[2] - EM_weight[predict_event][2])
                urow = abs(input_data[3] - EM_weight[predict_event][3])
                urge = np.array([ux , uy , uroz , urow])
                if(urge > 0.05).any():
                    error_num += 1
                    break





            else:
                EEM_weight = np.vstack((EEM_weight, input_list))
                print("new eposidic", EEM_weight.shape[0])
                #C1 = np.array([input_x, input_y, i])
                print("remember", i)
                #episodic_map.append(C1)
                activate_episodic[0 , 0] = input_x
                activate_episodic[0 , 1] = input_y
                break


    else:
        #新情景神经元编码完成，同时生成情景地图
        EEM_weight = np.vstack((EEM_weight, input_list))
        activate_episodic[0 , 0] = input_x
        activate_episodic[0 , 1] = input_y
        print("new eposidic" , EEM_weight.shape[0])

        #情景地图的生成

    #情景地图神经元整合
    #C = np.array([input_list , input_x , input_y , input_oz , input_ow , input_data])
    #C1 = np.array([input_x , input_y , i])
    print("remember" , i)
    episodic_map = np.vstack((episodic_map , activate_episodic))




    np.save("D:/project/corridor_data/episodic_map" , episodic_map)

    episodic_num.append(EEM_weight.shape[0])
    np.save("D:/project/corridor_data/episodic" , episodic_num)
    loop.append(loop_num)
    np.save("D:/project/corridor_data/loop", loop)
    event_num.append(EM_weight.shape[0])
    np.save("D:/project/corridor_data/event" , event_num)


print(EEM_weight.shape)
print(EM_weight.shape)
print(error_num)
print(loop_num)