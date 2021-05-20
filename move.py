import os
# from PIL import Image

import shutil


# path = 'E:/Lin/data_1208/third'
# new_path = 'E:/Lin/data_1208/test'

def moveImg(path, new_path):
    print(new_path)
    list1 = range(1, 22)
    for root, dirs, files in os.walk(path):
        print("长度-------" + str(len(files)))
        # print(files[1], type(files[1]))
        # print(type(root))
        if len(dirs) == 0:
            # print(len(dirs))
            for i in range(len(files)):
                # print("裁剪",str(files[i]).split('_')[-1])
                str1 = str(files[i]).split('_')[-1]  # 下划线后数字

                # print((str(list1[i])+'.jpg'))
                # print(type(str1[10:]))
                # number = filter(str.isdigit,  str1[10:]  )
                # print(int(number))
                number = re.findall(r'\d+', str1[10:])
                for j in range(21):  # 时间
                    if str1[10:] == (str(j) + '.jpg'):

                        print(files[i])
                        # print(number[0])
                        file_path = root + '/' + files[i]
                        new_file_path = new_path + '/' + str(j)
                        # os.mkdir( new_file_path)
                        my_file = Path(new_file_path)
                        if my_file.is_dir():
                            pass
                        else:
                            os.makedirs(new_file_path, 0o777)
                        # os.makedirs( new_file_path, 0o777 )
                        new_file_path = new_path + '/' + str(j) + '/' + files[i]
                        print("路径")  # +str(new_file_path = new_path+ '/'+ files[1]))
                        shutil.move(file_path, new_file_path)

    print("----end--------")


# """
# 将filePath文件下的图片保存在newFilePath文件夹下的相应子文件夹中
# pic 是字典，存放每个图片要移到的子文件夹名
# """
# def moveImg(filePath, newFilePath, pic):
#     # filePath = str(filePath, "utf8")
#     # newFilePath = str(newFilePath, "utf8")
#     for root, dirs, files in os.walk(filePath):
#         for f in files:
#             fl = filePath + '/' + f
#             img = Image.open(fl)
#             img.save(newFilePath + '/' +  pic[f[:-4]] + '/' + f) 


# C:\Users\Teng\Desktop\all_chong
if __name__ == "__main__":
    from pathlib import Path
    import re

    path = "C:/Users/xun/Desktop/all_chong"
    new_path = "C:/Users/xun/Desktop/move"
    my_file = Path(new_path)
    # print(my_file)
    if my_file.is_dir():
        pass
    else:
        os.makedirs(new_path, 0o777)
    # pic = 'C:/Users/xun/Desktop/move/move'
    # pic = 'C:\Users\xun\Desktop\all_chong\move\move'

    # moveImg(filePath, newFilePath, pic)
    moveImg(path, new_path)
