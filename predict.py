import torch
import models
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from train import MyDataset

classes = ('0', '1', '2', '3','4')

device = torch.device('cuda')


transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# 读图像

# val_path = r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\dataset\val\labels.txt'
# val_dataset = MyDataset(txt_path=val_path, transform=transform)
#
#
# val_dataloader = DataLoader(dataset=val_dataset,
#                               batch_size=4,
#                               shuffle=True,
#                               num_workers=4)
#  batch_size每次训练用的样本数量，shuffle打乱数据，num_workers是 mini-batch并行进程*4
test_path = 'E:\\empty\\c\\predict\\label_img.txt'
test_dataset = MyDataset(txt_path=test_path, transform=transform)


test_dataloader =  DataLoader(dataset=test_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=4)

'''
def predict(img_path):
    # net=torch.load('Lenet.pth') # pth格式 只保留参数
    # print('net', net)

    net = models.Classification()
    net.load_state_dict(torch.load('Lenet.pth'))
    net = net.to(device)
    torch.no_grad()
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    img_ = img.to(device)
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    # print(predicted)
    print('this picture maybe :', classes[predicted[0]])
    return classes[predicted[0]]
'''


def predict():
    # net=torch.load('Lenet.pth') # pth格式 只保留参数
    # print('net', net)

    acc = 0
    total = 0
    net = models.Classification()
    net.load_state_dict(torch.load('Lenet.pth'))
    net = net.to(device)
    with torch.no_grad():
        for index, data in enumerate(test_dataloader, start=0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, dim=1)
            print('number {} picture maybe :'.format(index), classes[predicted[0]])
            total += labels.size(0)
            acc += (predicted == labels).sum().item()
    print('Accuracy on test set : {}%'.format(100 * acc / total))




if __name__ == '__main__':
    # predict(r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\dataset\test\N_both_eyes_area_frame_48.jpg')
    # predict(r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\result\images\test\video2_far\concat\F_both_eyes_area_frame_300.jpg')
    # predict(r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\result\images\test\video2_near\concat\N_both_eyes_area_frame_30.jpg')
    '''
    result = []
    c = 0
    for i in range(300,501):
        if os.path.exists(r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\result\images\test\video2_far\concat\F_both_eyes_area_frame_{}.jpg'.format(i)):
            c += 1
            result.append(predict(
                r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\result\images\test\video2_far\concat\F_both_eyes_area_frame_{}.jpg'.format(i)))
        else:
            print('pic_{} is not exit'.format(i))
    print('acc = ', result.count('1')/c)  # 测试准确率为0.7846153846153846
    '''

    predict()
