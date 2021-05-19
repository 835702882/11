import json

with open('E:\\shujuji\\Data_ICRA18\\cloth_metadata.json', 'r', encoding='utf8')as fp:
    json_data = json.load(fp)


label_all = open('E:\\ccc\\新建文件夹\\label.txt', 'w', encoding='utf8')

with open('E:\\ccc\\all\\label_img.txt', 'r', encoding='utf8')as fp_label:
    for item in fp_label.readlines():
        # 107_26.jpg,1
        labels = json_data[str(item).split('_')[0]][:]
        # 0, 0, 0, 0, 0, 0, 0, 1
        image_name_labels = str(item).split(',')[0] + ',' + str(labels).replace('[', '').replace(']', '')
        label_all.write(image_name_labels + '\n')
        label_all.flush()

label_all.close()


# 这个程序可以挑选标签，11个标签可以选择 [0]][:]后面括号里的可以进行选择

