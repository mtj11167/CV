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
    scene_shape = scene_image.shape

    template_image = cv2.imread(template_path,cv2.IMREAD_GRAYSCALE)
    template_guassian = cv2.GaussianBlur(template_image, (5, 5), 0)
    template_canny = cv2.Canny(template_guassian, 100, 150)
    # template = template.astype(dtype=np.int32)
    template_shape = template_image.shape


    hausdorff_dis = {}
    try:
        start_all = time.time()
        for i in range(scene_shape[0] - template_shape[0]):
            start = time.time()
            for j in range(scene_shape[1] - template_shape[1]):
                start = time.time()
                scene_temp = scene_canny[i:i + template_shape[0], j:j + template_shape[1]]
                subset_scene_temp = scene_temp.nonzero()
                subset_template_temp = template_canny.nonzero()
                h_AB = np.zeros(shape=(subset_scene_temp[1].size,subset_template_temp[1].size),dtype=np.int32)
                for m in range(h_AB.shape[0]):
                    for n in range(h_AB.shape[1]):
                        x2=abs(subset_scene_temp[0][m]-subset_template_temp[0][n])
                        y2=abs(subset_scene_temp[1][m]-subset_template_temp[1][n])
                        h_AB[m,n] = x2+y2
                max_h_AB=np.max(np.min(h_AB,1))

                h_BA = np.zeros(shape=(subset_template_temp[1].size,subset_scene_temp[1].size),dtype=np.int32)
                for m in range(h_BA.shape[0]):
                    for n in range(h_BA.shape[1]):
                        x2 = abs(subset_scene_temp[0][n] - subset_template_temp[0][m])
                        y2 = abs(subset_scene_temp[1][n] - subset_template_temp[1][m])
                        h_BA[m, n] = x2 + y2
                max_h_BA = np.max(np.min(h_BA, 1))
                hausdorff_dis[(i,j)] = max(max_h_AB,max_h_BA)
                end =time.time()
                print("coordinate :",i,"over,cost ",end-start)
        end_all = time.time()
        print("all time", end_all - start_all)
    except:
        print("溢出")

     # 保存
    f = open('hausdorff_dis.txt', 'w')
    f.write(str(hausdorff_dis))
    f.close()

    # # 读取
    # f = open('temp.txt', 'r')
    # a = f.read()
    # dict_name = eval(a)
    # f.close()
    max_coo_show = min(zip(hausdorff_dis.values(), hausdorff_dis.keys()))
    print(max_coo_show)
    max_coo = (max_coo_show[1][1] + 1, scene_shape[0] - 1 - max_coo_show[1][0])
    print("最终坐标：", max_coo)

    rect = patches.Rectangle((max_coo_show[1][1], max_coo_show[1][0]), template_shape[1], template_shape[0],
                             linewidth=0.5, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    plt.savefig(template_path.split('.')[0].split('\\')[-1] + "_Hausdorff Distance.jpg")
    plt.show()

def DT_Hausdorff(scene_path,template_path):
    scene_image = cv2.imread(scene_path, cv2.IMREAD_GRAYSCALE)
    fig, ax1 = plt.subplots(1)
    plt.imshow(scene_image, cmap='gray')
    scene_guassian = cv2.GaussianBlur(scene_image, (5, 5), 0)
    # scene_canny = cv2.Laplacian(scene_guassian, -1,ksize=3)
    scene_canny = cv2.Canny(scene_guassian, 100, 150)
    scene_shape = scene_image.shape

    template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    template_guassian = cv2.GaussianBlur(template_image, (5, 5), 0)
    template_canny = cv2.Canny(template_guassian, 120, 150)
    template_shape = template_image.shape
    subset_template_temp = template_canny.nonzero()

    dt_template = np.full(shape=(template_canny.shape[0], template_canny.shape[1]),fill_value=200,dtype=np.int32)
    a = time.time()
    for m in range(dt_template.shape[0]):
        for n in range(dt_template.shape[1]):
            min_temp = 200
            for o in range(subset_template_temp[1].size):
                x2 = abs(subset_template_temp[0][o] - m )
                y2 = abs(subset_template_temp[1][o] - n )
                if (x2 + y2) < min_temp:
                    min_temp = x2 + y2
            dt_template[m, n] = min_temp
    # dt_dis_tempalte = np.min(dt_template, 2)
    # plt.imshow(dt_template,cmap='gray')
    # plt.show()
    b = time.time()
    print("aaaaa",b-a)

    hausdorff_dis = {}
    try:

        subset_scene_temp = scene_canny.nonzero()
        dt_scene = np.full(shape=(scene_canny.shape[0], scene_canny.shape[1]),fill_value=200, dtype=np.int32)
        start_all = time.time()
        for i in range(scene_shape[0]):
            start = time.time()
            for j in range(scene_shape[1]):
                min_temp = 200
                for o in range(subset_scene_temp[0].size):
                    x2 = abs(subset_scene_temp[0][o]-i)
                    y2 = abs(subset_scene_temp[1][o]-j)
                    if min_temp > x2 + y2:
                        min_temp = x2 + y2
                dt_scene[i, j] = min_temp
            end = time.time()
            print("coordinate :", i, "over,cost ", end - start)
        # plt.imshow(dt_scene, cmap='gray')
        # plt.show()
        for i in range(scene_shape[0] - template_shape[0]):
            for j in range(scene_shape[1] - template_shape[1]):
                dis = np.abs(dt_template - dt_scene[i:i+template_shape[0],j:j+template_shape[1]])
                hausdorff_dis[(i, j)] = np.sum(dis)
        end_all = time.time()
        print("all time", end_all - start_all)
    except:
        print("溢出")
    # 保存
    f = open('DT_hausdorff_dis.txt', 'w')
    f.write(str(hausdorff_dis))
    f.close()
    max_coo_show = min(zip(hausdorff_dis.values(), hausdorff_dis.keys()))
    print(max_coo_show)
    max_coo = (max_coo_show[1][1] + 1, scene_shape[0] - 1 - max_coo_show[1][0])
    print("最终坐标：", max_coo)

    rect = patches.Rectangle((max_coo_show[1][1], max_coo_show[1][0]), template_shape[1], template_shape[0],
                             linewidth=0.5, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    plt.savefig(template_path.split('.')[0].split('\\')[-1] + "_Distance Transform Hausdorff Distance.jpg")
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
            correlation_matching(os.path.join(os.getcwd(),"w3_data",scene),os.path.join(os.getcwd(),"w3_data",template))
            Hausdorff(os.path.join(os.getcwd(), "w3_data", scene),
                                 os.path.join(os.getcwd(), "w3_data", template))
            DT_Hausdorff(os.path.join(os.getcwd(), "w3_data", scene),
                                 os.path.join(os.getcwd(), "w3_data", template))
            print(template," over")

if __name__ ==  "__main__":
    all_pic()
