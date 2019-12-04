import os
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
import cv2
import time



def correlation_matching(scene_path,template_path):
    scene = mpimg.imread(scene_path)
    scene = scene.astype(dtype=np.int32)
    fig, ax1 = plt.subplots(1)
    plt.imshow(scene,cmap='gray')
    scene_shape = scene.shape

    template = mpimg.imread(template_path)
    template = template.astype(dtype=np.int32)
    template_shape = template.shape

    sims = {}
    try:
        for i in range(scene_shape[0]-template_shape[0]):
            for j in range(scene_shape[1]-template_shape[1]):
                sim = np.sum(np.multiply(scene[i:i+template_shape[0],j:j+template_shape[1]],template))/\
                        np.sqrt(np.sum(np.square(template)).astype(np.int64)*np.sum(np.square(scene[i:i+template_shape[0],j:j+template_shape[1]])).astype(np.int64))
                sims[(i,j)] = sim
    except:
        print("溢出")
    max_coo_show = max(zip(sims.values(),sims.keys()))
    print(max_coo_show)
    max_coo = (max_coo_show[1][1]+1,scene_shape[0]-1-max_coo_show[1][0])
    print("最终坐标：",max_coo)

    rect = patches.Rectangle((max_coo_show[1][1],max_coo_show[1][0]), template_shape[1], template_shape[0], linewidth=0.5, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    plt.savefig(template_path.split('.')[0].split('\\')[-1]+"_Correlation Matching.jpg")
    plt.show()

def Hausdorff(scene_path,template_path):
    scene_image = cv2.imread(scene_path,cv2.IMREAD_GRAYSCALE)
    fig, ax1 = plt.subplots(1)
    plt.imshow(scene_image, cmap='gray')
    scene_guassian = cv2.GaussianBlur(scene_image, (5, 5), 0)
    scene_canny = cv2.Canny(scene_guassian, 100, 150)
    # cv2.imshow("bb", scene_canny)
    # cv2.waitKey()
    # scene = scene.astype(dtype=np.int32)
    # fig, ax1 = plt.subplots(1)
    # plt.imshow(scene, cmap='gray')
    scene_shape = scene_image.shape

    template_image = cv2.imread(template_path,cv2.IMREAD_GRAYSCALE)
    template_guassian = cv2.GaussianBlur(template_image, (5, 5), 0)
    template_canny = cv2.Canny(template_guassian, 100, 150)
    # template = template.astype(dtype=np.int32)
    template_shape = template_image.shape
    # cv2.imshow("aa", template_canny)
    # cv2.waitKey()

    hausdorff_dis = {}
    try:
        for i in range(scene_shape[0] - template_shape[0]):
            start = time.time()
            for j in range(scene_shape[1] - template_shape[1]):
                scene_temp = scene_canny[i:i + template_shape[0], j:j + template_shape[1]]
                subset_scene_temp = scene_temp.nonzero()
                subset_template_temp = template_canny.nonzero()
                h_AB = np.zeros(shape=(subset_scene_temp[0].size,subset_template_temp[0].size))
                for m in range(h_AB.shape[0]):
                    for n in range(h_AB.shape[1]):
                        x2=pow((subset_scene_temp[0][m]-subset_template_temp[0][n]),2)
                        y2=pow((subset_scene_temp[1][m]-subset_template_temp[1][n]),2)
                        h_AB[m,n] = x2+y2
                max_h_AB=np.max(np.min(h_AB,1))

                h_BA = np.zeros(shape=(subset_template_temp[0].size,subset_scene_temp[0].size))
                for m in range(h_BA.shape[0]):
                    for n in range(h_BA.shape[1]):
                        x2 = pow((subset_scene_temp[0][n] - subset_template_temp[0][m]), 2)
                        y2 = pow((subset_scene_temp[1][n] - subset_template_temp[1][m]), 2)
                        h_BA[m, n] = x2 + y2
                max_h_BA = np.max(np.min(h_BA, 1))
                hausdorff_dis[(i,j)] = max(max_h_AB,max_h_BA)
            end =time.time()
            print("coordinate :",i,"over,cost ",end-start)
    except:
        print("溢出")

    max_coo_show = min(zip(hausdorff_dis.values(), hausdorff_dis.keys()))
    print(max_coo_show)
    max_coo = (max_coo_show[1][1] + 1, scene_shape[0] - 1 - max_coo_show[1][0])
    print("最终坐标：", max_coo)

    rect = patches.Rectangle((max_coo_show[1][1], max_coo_show[1][0]), template_shape[1], template_shape[0],
                             linewidth=0.5, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    plt.savefig(template_path.split('.')[0].split('\\')[-1] + "_Hausdorff Distance.jpg")
    plt.show()

def all_pic():
    '''
    调用所有图片
    :return:
    '''
    for root,dirs,files in os.walk("./w3_data"):
        templates = []
        scene = 0
        for file in files:
            if file.split('_')[0] == "Template":
                templates.append(file)
            else:
                scene =file
        for template in templates:
            # correlation_matching(os.path.join(os.getcwd(),"w3_data",scene),os.path.join(os.getcwd(),"w3_data",template))
            Hausdorff(os.path.join(os.getcwd(), "w3_data", scene),
                                 os.path.join(os.getcwd(), "w3_data", template))
            # image = cv2.imread(os.path.join(os.getcwd(),"w3_data",scene),cv2.IMREAD_GRAYSCALE)
            # sobel = cv2.Sobel(image,cv2.CV_16S,1,1,ksize =7)
            # sobel = cv2.convertScaleAbs(sobel)
            # cv2.imshow("aa",sobel)
            print(template," over")

if __name__ ==  "__main__":
    all_pic()
    a = np.asarray([[0,1,2,3,0,4,0],[0,1,2,3,0,4,0]])
    b=a.nonzero()
    print(a.ravel())
    print(np.flatnonzero(a))
    print(a.ravel()[np.flatnonzero(a)])
    print(a.nonzero())
    # a=(2,3)
    # b = (2, 3)
    # print(a == b)
    # print(np.sqrt(np.asarray([1,25,4])))
    # di = {(1,1):1,(4,1):-2,(1,2):3,(1,3):4}
    # min_a = min(zip(di.values(),di.keys()))
    # print(min_a)
    # a = np.asarray([160733043],dtype=np.int32)
    # b = np.asarray([101771192],dtype=np.int32)
    # try:
    #     print(np.asarray([np.sum(a),np.sum(b)],dtype=np.int64))
    #     print(np.sum(a).astype(dtype=np.int64)*np.sum(b).astype(dtype=np.int64))
    #     print(np.sum(a).astype(dtype=np.int64)*np.sum(b).astype(dtype=np.int64))
    # except:
    #     print("1111111")