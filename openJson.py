import json

temp = []
# 读取json文件内容,返回字典格式
with open('E:\\shujuji\\Data_ICRA18\\test_data.json', 'r', encoding='utf8')as fp:
    json_data = json.load(fp)

    for key in json_data.keys():
        temp.append(json_data[key]['cloth_index'])

temp.sort()

print(temp)

t1 = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]
print(len(t1))






