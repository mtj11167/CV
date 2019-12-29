import cv2
import os
import math
import time

def get_vedio():
    video_dir = './result.avi'  # 视频存储路径及视频名
    fps = 30  # 帧率一般选择20-30
    num = 201  # 图片数+1，因为后面的循环是从1开始
    img_size = (352,240)  # 图片尺寸，若和图片本身尺寸不匹配，输出视频是空的

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    for root,dirs,files in os.walk(os.getcwd()+"/mydata"):
        for file in files:
            frame = cv2.imread(os.getcwd()+"/mydata/"+file)
            videoWriter.write(frame)
            print(file)
    videoWriter.release()
    print('finish')

class OT(object):
    def __init__(self):
        self.has_object = False
        pass
    def start(self,path):
        for i in range(1,101):
            if i < 10:
                name = "car00"+str(i)+".bmp"
            elif i >= 10 and i <= 99:
                name = "car0" + str(i) + ".bmp"
            else:
                name = "car"+str(i) + ".bmp"
            image = cv2.imread(os.path.join(path, name))
            # image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # 用第一帧来标记框并且保存参数
            if not self.has_object:
                self.get_objecct(name,image)
                continue
            s_time  = time.time()
            self.meanshift(name,image)
        e_time =time.time()
        print("time:",e_time-s_time)
    def get_p(self,object):
        '''
        计算各种p或者q，
        :param object: 对应的灰度图像
        :return:
        '''
        h = pow(self.rect[3] / 2, 2) + pow(self.rect[2] / 2, 2)
        weight = [0] * (self.rect[3] * self.rect[2])
        p = [0] * 256
        C = 0
        for i in range(self.rect[3]):
            for j in range(self.rect[2]):
                x2 = pow(i - self.rect[3] / 2, 2) + pow(j - self.rect[2] / 2, 2)
                weight[i * self.rect[2] + j] = 1 - x2 / h
                C += weight[i * self.rect[2] + j]
                pixel = object[i, j]
                p[pixel] += weight[self.rect[2] * i + j]
        return p

    def get_objecct(self, name, img):
        '''
        对初始帧画目标框，并且计算保存权值以及位置大小等信息，对灰度图像操作
        :param name:
        :param img:
        :return:
        '''
        self.has_object = True
        rectt = cv2.selectROI("get Object", img)
        self.rect = [rectt[0], rectt[1], rectt[2], rectt[3]]
        image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        object = image_gray[int(self.rect[1]):int(self.rect[1] + self.rect[3]),
                 int(self.rect[0]):int(self.rect[0] + self.rect[2])]
        h = pow(self.rect[3] / 2, 2) + pow(self.rect[2] / 2, 2)
        self.weight = [0] * (self.rect[3] * self.rect[2])
        self.q = [0] * 256
        self.C = 0
        for i in range(self.rect[3]):
            for j in range(self.rect[2]):
                x2 = pow(i - self.rect[3] / 2, 2) + pow(j - self.rect[2] / 2, 2)
                self.weight[i * self.rect[2] + j] = 1 - x2/h
                self.C += self.weight[i * self.rect[2] + j]
                pixel = object[i, j]
                self.q[pixel] += self.weight[self.rect[2] * i + j]
        self.q = [i / self.C for i in self.q]

        img_save = cv2.rectangle(img, (self.rect[0], self.rect[1]),
                                 (self.rect[0] + self.rect[2], self.rect[1] + self.rect[3]), color=(0, 255, 0),
                                 thickness=1)
        cv2.imwrite("./mydata/" + name, img_save)

    def meanshift(self, name, img):
        '''
        按照课程的伪代码编写
        :param name:
        :param img:
        :return:
        '''
        image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        while True:
            object = image_gray[round(self.rect[1]):round(self.rect[1] + self.rect[3]),
                     round(self.rect[0]):round(self.rect[0] + self.rect[2])]
            # 计算p0
            self.p0 = [0]*256
            for i in range(self.rect[3]):
                for j in range(self.rect[2]):
                    self.p0[object[i,j]] +=self.weight[self.rect[2] * i + j]
            # 计算w
            self.w = [0]*256
            for i in range(256):
                if self.p0[i] != 0:
                    self.w[i] = math.sqrt(self.q[i] / self.p0[i])
            # 按照公式计算得到更新后的坐标
            x1 =0
            x2=0
            for i in range(self.rect[3]):
                for j  in range(self.rect[2]):
                    x1 = x1+self.w[object[i,j]]*(j-self.rect[2]/2)
                    x2 = x2+self.w[object[i,j]]*(i-self.rect[3]/2)
            y1 = x1/sum(self.w)
            y2 = x2/sum(self.w)

            y1 += self.rect[0]
            y2 += self.rect[1]
            # 计算p1
            self.p1 = [0] * 256
            for i in range(self.rect[3]):
                for j in range(self.rect[2]):
                    self.p1[image_gray[round(y2+i),round(y1+j)]] += self.weight[self.rect[2] * i + j]
            rou0 = sum([math.sqrt(self.q[i]*self.p0[i]) for i in range(256)])
            rou1 = sum([math.sqrt(self.q[i] * self.p1[i]) for i in range(256)])
            while rou1 < rou0:
                y1 = 0.5*(self.rect[0]+y1)
                y2 = 0.5*(self.rect[1]+y2)
                object_new = image_gray[round(y2):round(y2 + self.rect[3]),
                     round(y1):round(y1 + self.rect[2])]
                self.p1 = self.get_p(object_new)
                rou1 = sum([math.sqrt(self.q[i] * self.p1[i]) for i in range(256)])
            if (pow(y1-self.rect[0],2)+pow(y2-self.rect[1],2)) < 0.05:
                break
            else:
                self.rect[0] =y1
                self.rect[1] = y2
        img_save = cv2.rectangle(img, (round(self.rect[0]), round(self.rect[1])),
                                 (round(self.rect[0]) + self.rect[2], round(self.rect[1]) + self.rect[3]), color=(0, 255, 0),
                                 thickness=1)
        # cv2.imshow(name,img_save)
        # cv2.waitKey(30)
        cv2.imwrite("./mydata/" + name, img_save)

if __name__ == "__main__":
    path = os.path.join(os.getcwd(),"Car_Data")
    ot = OT()
    ot.start(path)
    get_vedio()