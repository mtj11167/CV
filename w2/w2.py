import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np


def all_pic():
    '''
    调用所有图片
    :return:
    '''
    for root,dirs,files in os.walk("./w2_data"):
        for file in files:
            k=int(file.split('.')[0].split('_')[2])+1
            vector_length = 3
            for i in range(5,53,2):
                image = pic(os.getcwd()+"/w2_data/"+file,k,vector_length,i)
                plt.imshow(image,'gray')
                plt.savefig(file.split('.')[0]+"_"+str(vector_length)+"_"+str(i)+".jpg")
                print(file.split('.')[0]+"_"+str(vector_length)+"_"+str(i)+".jpg"+" over")
            print(file,"over")

def get_template(scale):
    '''
    根据提供模板大小计算对应模板坐标
    比如大小为3，得到[(-1-,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    :param scale:模板大小
    :return:
    '''
    ret =[]
    center = scale // 2
    for i in range(scale):
        for j in range(scale):
            if i == center and j == center:
                continue
            ret.append((i-center,j-center))
    return ret

def get_feature(image,template,x,y):
    '''
    根据模板得到某个点的特征
    :param image: numpy.ndarray 图像
    :param template: 模板
    :param x:
    :param y:
    :return:
    '''
    feature = []
    for pos in template:
        if x+pos[0] < 0 or y+pos[1] < 0 or x+pos[0] >= image.shape[0] or y+pos[1] >= image.shape[1]:
            feature.append(0)
        elif image[x+pos[0],y+pos[1]] <= image[x,y]:
            feature.append(0)
        else:
            feature.append(1)
    ret = ''
    for bit in feature:
        ret+=str(bit)
    ret = int(ret,2)
    return ret

def get_histogram(vectors,scale_region,x,y,vector_length):
    '''
    用一定范围的特征得到某个点的直方图特征
    :param vectors:
    :param scale_region:
    :param x:
    :param y:
    :param vector_length:
    :return:
    '''
    template_region = get_template(scale_region)
    features = []
    miss_counter = 0
    for pos in template_region:
        if x+pos[0] < 0 or y+pos[1] < 0 or x+pos[0] >= vectors.shape[0] or y+pos[1] >= vectors.shape[1]:
            miss_counter +=1
            continue
        features.append(vectors[x+pos[0],y+pos[1]])
    freq = Counter(features)
    ret = np.zeros([1,vector_length])
    for pair in freq.items():
        ret[0,pair[0]-1] = pair[1]
    # 假设均匀分布，将统计的数量翻倍
    ret = ret*(1+miss_counter/(len(template_region)-miss_counter))
    ret.astype(dtype=np.int32)
    return ret

def pic(path,k,scale_template,scale_region):
    '''
        根据路径计算每一个图片的值
        :param path:图片路径
        :return:
        '''
    image = mpimg.imread(path)
    # 得到每个点的特征
    shape = image.shape
    feature_template = get_template(scale_template)
    vectors = np.zeros(shape=(shape[0],shape[1]),dtype=np.int32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            vectors[i,j]=get_feature(image,feature_template,i,j)
    # 得到每个点的直方图特征
    features_histogram = np.zeros(shape=(shape[0]*shape[1],pow(2,len(feature_template))))
    for i in range(shape[0]):
        for j in range(shape[1]):
            features_histogram[i*shape[0]+j,:]=get_histogram(vectors,scale_region,i,j,pow(2,len(feature_template)))
    # 聚类
    y=KMeans(k).fit_predict(features_histogram)
    y.reshape((y.shape[0],1))
    #生成效果图
    image_show=[rank*(255//(k-1)) for rank in y]
    image_show=np.asarray(image_show)
    image_show=np.reshape(image_show,newshape=(image_show.shape[0],1))
    image_show=np.reshape(image_show,newshape=(shape[0],shape[1]))

    return image_show

if __name__ == "__main__":
    all_pic()