import cv2
import os
import math

class ObejectTracking(object):
    def __init__(self):
        self.has_object = False

    def get_objecct(self,name,img):
        self.has_object = True
        rectt = cv2.selectROI("get Object", img)
        self.rect = [rectt[0],rectt[1],rectt[2],rectt[3]]
        image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        object = image_gray[int(self.rect[1]):int(self.rect[1] + self.rect[3]), int(self.rect[0]):int(self.rect[0] + self.rect[2])]
        h = pow(self.rect[3]/2,2)+pow(self.rect[2]/2,2)
        self.weight1 = [0]*(self.rect[3]*self.rect[2])
        self.histogram1 = [0]*256
        self.C = 0
        for i in range(self.rect[3]):
            for j in range(self.rect[2]):
                temp = pow(i-self.rect[3]/2,2)+pow(j-self.rect[2]/2,2)
                self.weight1[i*self.rect[2]+j] = 1-temp/h
                self.C+=self.weight1[i*self.rect[2]+j]
        for i in range(self.rect[3]):
            for j in range(self.rect[2]):
                pixel = object[i,j]
                self.histogram1[pixel]+=self.weight1[self.rect[2]*i+j]
        self.histogram1 = [i/self.C for i in self.histogram1]

        img_save = cv2.rectangle(img,(self.rect[0],self.rect[1]),(self.rect[0]+self.rect[2],self.rect[1]+self.rect[3]),color=(0,255,0),thickness=1)
        cv2.imwrite("./mydata/"+name,img_save)

    def meanshift(self,name,img):
        image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        y1=2
        y2=2
        num =0
        stopNum = 20
        h = pow(self.rect[3] / 2, 2) + pow(self.rect[2] / 2, 2)

        while (pow(y1,2)+pow(y2,2) > 0.5 and num < stopNum):
            self.histogram2 = [0] * 256
            self.w = [0]*256
            self.weight2 = [0] * (self.rect[3] * self.rect[2])
            pixell = [0]*(self.rect[3] * self.rect[2])
            # C = 0
            for i in range(self.rect[3]):
                for j in range(self.rect[2]):
                    pixell[i*self.rect[2]+j] = image_gray[i+round(self.rect[1]),j+round(self.rect[0])]
                    # temp = pow(i - self.rect[3] / 2, 2) + pow(i - self.rect[2] / 2, 2)
                    # self.weight2[i * self.rect[2] + j] = 1 - temp / h
                    # C += self.weight2[i * self.rect[2] + j]
                    self.histogram2[pixell[i*self.rect[2]+j]] += self.weight1[i*self.rect[2]+j]
            self.histogram2 = [i/self.C for i in self.histogram2]
            for i in range(256):
                if self.histogram2[i] != 0:
                    self.w[i] = math.sqrt(self.histogram1[i]/self.histogram2[i])
            sum_w =0
            x1=0
            x2=0
            for i in range(self.rect[3]):
                for j in range(self.rect[2]):
                    sum_w += self.w[pixell[i*self.rect[2]+j]]
                    x1+= self.w[pixell[i*self.rect[2]+j]]*(i-self.rect[3]/2)
                    x2 += self.w[pixell[i * self.rect[2] + j]] * (j - self.rect[2] / 2)
            y1 = x1/sum_w
            y2 = x2/sum_w

            # self.rect[0]= int(self.rect[0]+y2)
            # self.rect[1]= int(self.rect[1]+y1)
            self.rect[0] += y2
            self.rect[1] += y1
        cv2.rectangle(img, (round(self.rect[0]), round(self.rect[1])), (round(self.rect[0]) + self.rect[2], round(self.rect[1]) + self.rect[3]),
                      color=(0, 255, 0), thickness=1)
        cv2.imshow(name,img)
        cv2.waitKey(30)
        cv2.imwrite("./mydata/" + name, img)

    def start(self,path):
        for i in range(1,101):
            if i < 10:
                name = "car00"+str(i)+".bmp"
            elif i >= 10 and i <= 99:
                name = "car0" + str(i) + ".bmp"
            else:
                name = "car"+str(i) + ".bmp"
            image = cv2.imread(os.path.join(path, name))
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if not self.has_object:
                self.get_objecct(name,image)
                continue
            self.meanshift(name,image)




if __name__ == "__main__":
    path = os.path.join(os.getcwd(),"Car_Data")
    ot = ObejectTracking()
    ot.start(path)