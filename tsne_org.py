
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import os
import cv2
def get_data():
    labels = []
    pic_np_list = []
    # pic_list = os.listdir(r'D:\Postgraduate-Program\smart-wheelchair\dataset\origin_data')
    folder_list = os.listdir(r'E:\new\Test5_resnet\data_set\flower_data\flower_photos')
    for folder_name in folder_list:
        # if pic == 'labels.txt':
        #     continue

        # label = pic.split('.')[0].split('_')[0]

        labels.append(int(folder_name))
        pic_list = os.listdir(os.path.join(r'E:\new\Test5_resnet\data_set\flower_data\flower_photos',folder_name))
        for pic in pic_list:
            pic_np = cv2.imread(os.path.join(r'E:\new\Test5_resnet\data_set\flower_data\flower_photos', folder_name, pic))
            # print(pic_np)

            pic_np_list.append(pic_np.reshape(1,224*224*3))
    print(len(pic_np_list))
    a = np.concatenate(pic_np_list)



    labels = np.array(labels)

    return a, labels


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    print(data.shape[0])
    for i in range(data.shape[0]):

        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def main():
    data, label = get_data()
    print(len(data))
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.savefig('data distribution.jpg')
    plt.show()

if __name__ == '__main__':
    main()

