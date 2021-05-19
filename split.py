# import os
# import cv2
#
#
# def get_img_list(dir, firelist, ext=None):
#     newdir = dir
#     if os.path.isfile(dir):  # 如果是文件
#         if ext is None:
#             firelist.append(dir)
#         elif ext in dir[-3:]:
#             firelist.append(dir)
#     elif os.path.isdir(dir):  # 如果是目录
#         for s in os.listdir(dir):
#             newdir = os.path.join(dir, s)
#             get_img_list(newdir, firelist, ext)
#
#     return firelist
#
#
# def read_img():
#     image_path = './all_chong'
#     imglist = get_img_list(image_path, [], 'jpg')
#     imgall = []
#     for imgpath in imglist:
#         print(imgpath)
#         imaname = os.path.split(imgpath)[1][15]  # 分离文件路径和文件名后获取文件名（包括了后缀名）
#         print(imaname)
#         if imaname == 0

#         img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
#         imgall.append(img)
#         cv2.namedWindow(imaname, cv2.WINDOW_AUTOSIZE)
#         cv2.imshow(imaname, img)
#     cv2.waitKey(0)
#
#     return imgall
#
#
# if __name__ == '__main__':
#     imgall = read_img()
#     print(imgall.__len__())


import json


train = set([])
with open('E:\\shujuji\\Data_ICRA18\\train_data.json', 'r', encoding='utf8')as fp:
    json_data = json.load(fp)
    for key in json_data.keys():
        train.add(json_data[key]['cloth_index'])
print(train)

test = set([])
with open('E:\\shujuji\\Data_ICRA18\\test_data.json', 'r', encoding='utf8')as fp:
    json_data = json.load(fp)
    for key in json_data.keys():
        test.add(json_data[key]['cloth_index'])

print(test)

train_label = open('E:\\empty\\big_resolution\\train_label.txt', 'w', encoding='utf8')
test_label = open('E:\\empty\\big_resolution\\test_label.txt', 'w', encoding='utf8')

with open('E:\\empty\\big_resolution\\label.txt', 'r', encoding='utf8')as fp_label:
    for item in fp_label.readlines():
        img_index = str(item).split('_')[0]
        if train.__contains__(int(img_index)):
            train_label.write(item)
            train_label.flush()
            continue
        test_label.write(item)
        test_label.flush()

train_label.close()
test_label.close()

