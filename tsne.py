
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import os
import cv2
def get_data():
    labels = []
    pic_np_list = []
    pic_list = os.listdir(r'E:\ccc\paper-data_set\flower_data\task10-6classes\test'
                          r'')
    for pic in pic_list:
        if pic == 'labels.txt':
            continue
        print(pic)
        # label = pic.split('')[0].split("_")[-1]
        label = pic.split('_')[2][9]
        print(label)

        labels.append(int(label))
        pic_np = cv2.imread(os.path.join(r'E:\ccc\paper-data_set\flower_data\task10-6classes\test', pic))
        print(os.path.join(r'E:\ccc\paper-data_set\flower_data\task10-6classes\test', pic))
        pic_np_list.append(pic_np.reshape(1,224*224*3))
    a = np.concatenate(pic_np_list)



    labels = np.array(labels)

    return a, labels


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
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
    plt.savefig('data distribution10.png')
    plt.show()

if __name__ == '__main__':
    main()

