import cv2
import os
import math
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def gaussian(x,u,sigma):
    return (1/(sigma*pow(2*math.pi,0.5)))*(math.exp(-(x-u)**2/(2*sigma**2)))

class GMM(object):
    def __init__(self,gassian_number = 7 ,row = 576,column =768,sigma = 7,alpha = 0.1,threshod = 0.6):
        # 使用灰度图
        self.gaussian_number = gassian_number
        self.row = row
        self.column = column
        self.is_first = False
        self.sigma = sigma
        self.alpha = alpha
        self.fit_num = np.ones(shape=(self.row,self.column),dtype=np.int32)
        self.threshod = threshod

    def init(self,image):
        # self.gaussian_para最后一维 0表示w，1表示均值，2表示方差
        self.gaussian_para = np.zeros(shape=(self.row, self.column, self.gaussian_number, 3))
        for i in range(image.shape[0]):
            for j  in range(image.shape[1]):
                for k in range(self.gaussian_number):
                    if k == 0:
                        self.gaussian_para[i,j,0,0] = 1
                        self.gaussian_para[i, j, 0, 1] = image[i,j]
                        self.gaussian_para[i, j, 0, 2] = self.sigma
                    else:
                        self.gaussian_para[i,j,k,2] = self.sigma

    def train(self,img):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                unfit = 0
                for k in range(self.gaussian_number):
                    if self.gaussian_para[i,j,k,0] != 0:
                        # 满足或者不满足某个高斯分布，需要更新参数
                        dis  = abs(img[i,j]-self.gaussian_para[i,j,k,1])
                        if dis*dis < 3*self.sigma:
                            self.gaussian_para[i,j,k,0] = self.alpha+(1-self.alpha)*self.gaussian_para[i,j,k,0]
                            rou = self.alpha*gaussian(img[i,j],self.gaussian_para[i,j,k,1],self.gaussian_para[i,j,k,2])
                            self.gaussian_para[i,j,k,1] = (1-rou)*self.gaussian_para[i,j,k,1]+rou*img[i,j]
                            self.gaussian_para[i,j,k,2] = math.sqrt((1-rou)*self.gaussian_para[i,j,k,2]**2+rou*(img[i,j]-self.gaussian_para[i,j,k,1])**2)
                        else:
                            self.gaussian_para[i, j, k, 0] = (1 - self.alpha) * self.gaussian_para[i, j, k, 0]
                            unfit+=1
                # 排序，使用冒泡算法,从大到小
                for m in range(self.gaussian_number):
                    for n in range(m,self.gaussian_number):
                        if self.gaussian_para[i,j,n,0]/self.gaussian_para[i,j,n,2] > self.gaussian_para[i,j,m,0]/self.gaussian_para[i,j,m,2]:
                            for index in range(3):
                                tempw = self.gaussian_para[i,j,n,index]
                                self.gaussian_para[i, j, n, index]  = self.gaussian_para[i,j,m,index]
                                self.gaussian_para[i, j, m, index] = tempw

                gau_now = 0
                for k in range(self.gaussian_number):
                    if self.gaussian_para[i,j,k,0] != 0:
                        gau_now+=1
                    else:
                        break
                # 如果都不满足而且现在高斯数目比最大数目小，建一个新的
                if unfit == gau_now and self.gaussian_para[i,j,self.gaussian_number-1,0] == 0:
                    for k in range(self.gaussian_number):
                        if self.gaussian_para[i,j,k,0] == 0:
                            self.gaussian_para[i,j,k,0] = self.alpha
                            self.gaussian_para[i,j,k,1] = img[i,j]
                            self.gaussian_para[i,j,k,2] = self.sigma
                            break
                elif unfit == gau_now and self.gaussian_para[i,j,self.gaussian_number-1,0] != 0:
                    self.gaussian_para[i,j,self.gaussian_number-1,1] = img[i,j]
                    self.gaussian_para[i,j,self.gaussian_number-1,2] = 15
                # 归一化
                sum_gui = 0
                for k in range(self.gaussian_number):
                    sum_gui+=self.gaussian_para[i,j,k,0]
                for k in range(self.gaussian_number):
                    self.gaussian_para[i,j,k,0] /= sum_gui

    def get_fit_num(self):
        for i in range(self.row):
            for j in range(self.column):
                sum = 0
                for k in range(self.gaussian_number):
                    sum+=self.gaussian_para[i,j,k,0]
                    if sum >= self.threshod:
                        self.fit_num[i,j] = k+1
                        break

    def gmm_test(self,img):
        ret = np.zeros(shape=(img.shape[0],img.shape[1]))
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if i == 410 and j == 710:
                    print("a")
                flag = False
                for k in range(self.fit_num[i,j]):
                    if abs(self.gaussian_para[i,j,k,1] - img[i,j]) < 2.5*self.gaussian_para[i,j,k,2]:
                        ret[i,j] = 0
                        flag =True
                        break
                if not flag:
                    ret[i, j] = 255
        return ret


    def start(self, path,train_start,train_end,test_end):
        for i in range(train_start,train_end):
            name = 0
            if i == 0:
                name = "0000.jpg"
            elif i > 0 and i<10:
                name = "000"+str(i)+".jpg"
            elif i >= 10 and i < 100:
                name = "00"+str(i)+".jpg"
            else:
                name = "0" + str(i) + ".jpg"
            image = cv2.imread(os.path.join(path, name))
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if self.is_first == False:
                self.init(image_gray)
                self.is_first = True
                continue
            start = time.time()
            self.train(image_gray)
            # cv2.imshow("1", image_gray)
            # cv2.waitKey()
            end = time.time()
            print(i," time", end - start)
        self.get_fit_num()
        np.save("./guapara_sigma"+str(self.sigma)+"_alpha"+str(self.alpha)+".npy",self.gaussian_para)
        for j in range(train_start,test_end):
            name = 0
            if j == 0:
                name = "0000.jpg"
            elif j > 0 and j < 10:
                name = "000" + str(j) + ".jpg"
            elif j >= 10 and j < 100:
                name = "00" + str(j) + ".jpg"
            else:
                name = "0" + str(j) + ".jpg"
            image = cv2.imread(os.path.join(path, name))
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            img_show = self.gmm_test(image_gray)
            cv2.imwrite("./mydata/" + name, img_show)

    def start_exist(self,path,train_end,test_end):
        self.gaussian_para = np.load("./guapara.npy")
        self.get_fit_num()
        for j in range(train_end,test_end):
            name = 0
            if j < 100:
                name = "00" + str(j) + ".jpg"
            else:
                name = "0" + str(j) + ".jpg"
            image = cv2.imread(os.path.join(path, name))
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            a = time.time()
            img_show = self.gmm_test(image_gray)
            b= time.time()
            print("time:",b-a)
            cv2.imwrite("./mydata/" + name, img_show)


if __name__ == "__main__":
    model = GMM(4,576,768)
    model.start(os.path.join(os.getcwd(), "Scene_Data"),0,110,200)
    # model.start_exist(os.path.join(os.getcwd(), "Scene_Data"),110,121)