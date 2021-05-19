import torch
import torch.nn as nn
import torch.optim as optim
import time
import models
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image
from torch.utils.data import DataLoader



class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r', encoding='utf-8')
        imgs = []
        for line in fh:
            line = line.rstrip()  # 去除txt文档中每行结尾的指定字符（默认是空格）
            words = line.split(',')    # 将每行数据以空格分开
            imgs.append((words[0], int(words[1]))) # 将图片地址 和对应的label以元组的形式放在一个列表中
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]        # 将存储在列表中的图像地址 和 label 分开
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# data_transform = transforms.Compose([
#     transforms.Resize((320, 36)),
#     # transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])

# train_dataset = datasets.ImageFolder(root='/Users/***/Tooth-Detection/data/train',transform=data_transform)
# train_dataset = datasets.ImageFolder(root=r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\dataset\train', transform=transform)
# train_dataset = datasets.ImageFolder(root=r'D:\1111', transform=transform)

train_path = 'E:\\empty\\c\\img\\label_img.txt'
train_dataset = MyDataset(txt_path=train_path, transform=transform)


train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=4)
#  batch_size每次训练用的样本数量4个一组，shuffle打乱数据，num_workers是 mini-batch并行进程*4
#
# val_path = r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\dataset\val\labels.txt'
# val_dataset = MyDataset(txt_path=val_path, transform=transform)
#
#
# val_dataloader = DataLoader(dataset=val_dataset,
#                               batch_size=4,
#                               shuffle=True,
#                               num_workers=4)



# val_dataset = datasets.ImageFolder(root=r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\dataset\val', transform=transform)
# val_dataloader = DataLoader(dataset=val_dataset,
#                               batch_size=4,
#                               shuffle=True,
#                               num_workers=4)
#
# test_dataset = datasets.ImageFolder(root=r'D:\研究生期间项目资料\Gaze-Estimation\可行性实验程序\dataset\test', transform=transform)
# test_dataloader = DataLoader(dataset=test_dataset,
#                               batch_size=4,
#                               shuffle=True,
#                               num_workers=4)


# 在windows下训练代码不在 __name == '__main__'的话 就会报错说The "freeze_support()" line can be omitted if the program
#         is not going to be frozen to produce an executable.
#https://discuss.pytorch.org/t/cant-iter-dataloader-object-brokenpipeerror/18451

# if __name__ == '__main__':
# print('Debug2', train_dataloader, iter(train_dataloader))
# test_data_iter = iter(train_dataloader)
# test_image, test_label = test_data_iter.next()
# print('Debug',test_image, test_label)

if __name__ == '__main__':
    net = models.Classification()  # 定义训练的网络模型
    # net = net.cuda()
    loss_function = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器（训练参数，学习率）

    for epoch in range(20):  # 一个epoch即对整个训练集进行一次训练
        running_loss = 0.0
        time_start = time.perf_counter()

        for step, data in enumerate(train_dataloader, start=0):  # 遍历训练集，step从0开始计算
            inputs, labels = data  # 获取训练集的图像和标签

            optimizer.zero_grad()  # 清除历史梯度

            # forward + backward + optimize
            # outputs = net(inputs.permute(0,1,3,2))  # 正向传播
            outputs = net(inputs)  # 正向传播
            loss = loss_function(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新参数

            # print statistics
            running_loss += loss.item()
            if step % 4 == 3:  # print every 4 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / 4))
                running_loss = 0.0


            # # 打印耗时、损失、准确率等数据
            # running_loss += loss.item()
            # if step % 1000 == 999:  # print every 1000 mini-batches，每1000步打印一次
            #     with torch.no_grad():  # 在以下步骤中（验证过程中）不用计算每个节点的损失梯度，防止内存占用
            #         outputs = net(
            #             test_image)  # 测试集传入网络（test_batch_size=10000），output维度为[10000,10]
            #         predict_y = torch.max(outputs, dim=1)[
            #             1]  # 以output中值最大位置对应的索引（标签）作为预测输出
            #         accuracy = (
            #                                predict_y == test_label).sum().item() / test_label.size(
            #             0)
            #
            #         print(
            #             '[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
            #             (epoch + 1, step + 1, running_loss / 500, accuracy))
            #
            #         print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
            #         running_loss = 0.0

    print('Finished Training')

    # 保存训练得到的参数
    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


