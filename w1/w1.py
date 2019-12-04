import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def all_pic():
    '''
    调用所有图片
    :return:
    '''
    for root,dirs,files in os.walk("./w1_data"):
        for file in files:
            pic(os.getcwd()+"/w1_data/"+file)
            print(file,"over")

def mix_guassian(u0,sigma0,p0,u1,sigma1,p1,x):
    '''
    混合高斯模型
    :param u0: float 第一个高斯函数的均值
    :param sigma0: float 第一个高斯函数的方差
    :param p0: float 第一个高斯函数的概率
    :param u1: float 第二个高斯函数的均值
    :param sigma1: float 第二个高斯函数的方差
    :param p1: float 第二个高斯函数的概率
    :param x: int 输入的像素值
    :return: float 返回输出的y值
    '''
    guassian_0=1/(np.power(2*np.pi,0.5)*sigma0)*np.exp(-np.power((u0-x),2)/(2*sigma0*sigma0))
    guassian_1 = 1 / (np.power(2 * np.pi, 0.5) * sigma1) * np.exp(-np.power((u1 - x), 2) / (2 * sigma1 * sigma1))
    return guassian_0*p0+guassian_1*p1

def cal_mean(frequency, index):
    '''
    计算均值
    :param frequency:numpy.ndarray 像素的频次
    :param index: int 像素的起始值
    :return: float 均值
    '''
    sum=0
    cunnter =0
    for i in range(frequency.shape[0]):
        sum+=frequency[i]*(index+i)
        cunnter+=frequency[i]
    return sum/cunnter

def cal_std(frequency, index):
    '''
    计算标准差
    :param frequency: numpy.ndarray 像素的频次
    :param index: int 像素的起始值
    :return:  float 标准差
    '''
    data= []
    for i in range(frequency.shape[0]):
        for j in range(frequency[i]):
            data.append(index+i)
    # 处理[0 0 0 0 0 0 2]的极端情况
    if len(set(data)) == 1:
        return 0.1
    return np.std(np.asarray(data))

def pic(path):
    '''
    根据路径计算每一个图片的值
    :param path:图片路径
    :return:
    '''
    image = mpimg.imread(path)
    # plt.imshow(image,cmap='gray')
    # plt.show()
    shape = image.shape
    # 计算最大像素值与最小像素值，减小计算量
    min_pixel = int(np.min(image))
    max_pixel = int(np.max(image))
    t = (max_pixel+min_pixel)//2
    # 以一定阈值的范围生成所有阈值
    t_set =[min_pixel+i for i in range(10,(max_pixel-min_pixel-10),1)]
    # t_set = [t + i for i in range(-(t-min_pixel)//2, (max_pixel - t)//2, 2)]
    # t_set=[i for i in range(80,150,5)]
    pixel_cnt =[0]*(max_pixel-min_pixel+1)
    pixel_cnt = np.asarray(pixel_cnt)
    # 统计像素频次，存入pixel_cnt
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixel_cnt[image[i,j]-min_pixel]+=1
    pixel_p = pixel_cnt/np.sum(pixel_cnt)


    k_set=[]
    min_t = 0
    min_k = 1000000000
    for t_temp in t_set:
        # 计算均值方差以及概率
        u_o=cal_mean(pixel_cnt[:t_temp-min_pixel],min_pixel)
        u_b=cal_mean(pixel_cnt[t_temp-min_pixel:],t_temp)
        sigma_o=cal_std(pixel_cnt[:t_temp-min_pixel],min_pixel)
        sigma_b = cal_std(pixel_cnt[t_temp-min_pixel:],t_temp)
        p_0 = np.sum(pixel_cnt[:t_temp-min_pixel])/np.sum(pixel_cnt[:])
        p_1 = 1-p_0
        # 显示每个阈值对应二值图
        # print("t_temp:",t_temp)
        # image_seg = np.zeros(shape=(shape[0], shape[1]))
        # for i in range(shape[0]):
        #     for j in range(shape[1]):
        #         if image[i, j] > t_temp:
        #             image_seg[i, j] = 255
        #         else:
        #             image_seg[i, j] = 0
        # plt.imshow(image_seg, 'gray')
        # plt.show()

        # 计算交叉熵
        k=0
        for i in range(pixel_cnt.shape[0]):
            if pixel_cnt[i] == 0:
                continue
            k+=pixel_p[i]*np.log(pixel_p[i]/(mix_guassian(u_o,sigma_o,p_0,u_b,sigma_b,p_1,min_pixel+i)))
        k_set.append(k)
        if min_k > k:
            min_k = k
            min_t = t_temp
    print("min t:",min_t," min k:",min_k)
    # 计算最优阈值对应均值方差,可优化
    x = [min_pixel + i for i in range(max_pixel - min_pixel + 1)]
    x_guassian = []
    u0_show = cal_mean(pixel_cnt[:min_t - min_pixel], min_pixel)
    ub_show = cal_mean(pixel_cnt[min_t - min_pixel:], min_t)
    sigmao_show = cal_std(pixel_cnt[:min_t - min_pixel],min_pixel)
    sigmab_show = cal_std(pixel_cnt[min_t - min_pixel:],min_t)
    p0_show = np.sum(pixel_cnt[:min_t - min_pixel]) / np.sum(pixel_cnt[:])
    p1_show = 1 - p0_show
    for i in range(pixel_cnt.shape[0]):
        x_guassian.append(50000*mix_guassian(u0_show, sigmao_show, p0_show, ub_show, sigmab_show, p1_show, min_pixel + i))
    plt.subplot(1,2,1)
    plt.plot(x, pixel_cnt)
    plt.plot(x,x_guassian,color="red")
    plt.subplot(1, 2, 2)
    plt.plot(t_set, k_set)
    plt.show()
    # 显示最佳二值图
    image_seg = np.zeros(shape=(shape[0],shape[1]))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if image[i,j] > min_t:
                image_seg[i,j] = 255
            else:
                image_seg[i, j] = 0
    plt.imshow(image_seg,'gray')
    plt.show()

if __name__ == "__main__":
    all_pic()
