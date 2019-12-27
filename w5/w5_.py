import cv2
import os

def get_vedio():
    video_dir = './a.avi'  # 视频存储路径及视频名
    fps = 30  # 帧率一般选择20-30
    num = 201  # 图片数+1，因为后面的循环是从1开始
    img_size = (768,576)  # 图片尺寸，若和图片本身尺寸不匹配，输出视频是空的

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    for root,dirs,files in os.walk(os.getcwd()+"/Scene_Data"):
        for file in files:
            frame = cv2.imread(os.getcwd()+"/Scene_Data/"+file)
            videoWriter.write(frame)
            print(file)
    videoWriter.release()
    print('finish')

def train():
    cap = cv2.VideoCapture('./a.avi')
    # knn_sub = cv2.createBackgroundSubtractorKNN()
    mog2_sub = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mog_sub_mask = mog2_sub.apply(frame)
        # knn_sub_mask = knn_sub.apply(frame)

        cv2.imshow('original', frame)
        cv2.imshow('MOG2', mog_sub_mask)
        # cv2.imshow('KNN', knn_sub_mask)

        key = cv2.waitKey(30) & 0xff
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__  == "__main__":
    pass
    # get_vedio()
    train()