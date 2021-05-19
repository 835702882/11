import os 
import shutil
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

data_path = 'E:\\empty\\big_resolution'
     # 路径为所有图片路径
for root, dirs, files in os.walk(data_path):
    for file in files:
        if train.__contains__(int(file.split('_')[0])):
            old_file_path = os.path.join(root, file)
            #print(old_file_path)
            new_path = 'E:\\empty\\img\\train'
            # 路径为训练图片路径
            if not os.path.exists(new_path):  # 创建新文件夹
                os.makedirs(new_path)
            new_file_path = new_path + '\\' + file
            print(new_file_path)
            shutil.copyfile(old_file_path, new_file_path) # 复制文件
print('finished!')
  # 本程序利用train和test.json 将所有图片划分成测试和训练
