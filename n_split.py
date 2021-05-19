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

train_label = open('E:\\ccc\\train\\train_label.txt', 'w', encoding='utf8')
test_label = open('E:\\ccc\\test\\test_label.txt', 'w', encoding='utf8')

with open('E:\\ccc\\all_chong\\label_img.txt', 'r', encoding='utf8')as fp_label:
    for item in fp_label.readlines():
        img_index = str(item).split('_')[0]
        print("img_index")
        if train.__contains__(int(img_index)):
            train_label.write(item)
            train_label.flush()
            continue
        test_label.write(item)
        test_label.flush()

train_label.close()
test_label.close()

