import os
import cv2
# 要提取视频的文件名，隐藏后缀
sourceFileName = 'E:\\shujuji\\Data_ICRA18\\Data\\11\\3\\GelSight_video'
# 在这里把后缀接上
video_path = os.path.join("", "", sourceFileName+'.mp4')
times = 0
# 提取视频的频率，每25帧提取一个

# 输出图片到当前目录vedio文件夹下
outPutDirName = 'E:\\empty\\c\\victory'
if not os.path.exists(outPutDirName):
    # 如果文件目录不存在则创建目录
    os.makedirs(outPutDirName)
camera = cv2.VideoCapture(video_path)
# camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
frameFrequency = int(camera.get(7))

while True:
    times += 1
    res, image = camera.read()
    if not res:
        print('not res , not image')
        break
    if times % frameFrequency == 0:
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(outPutDirName + str(times)+'.jpg', image)
        print(outPutDirName + str(times)+'.jpg')
print('图片提取结束')
camera.release()