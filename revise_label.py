import os
label_file = open('dataset/label_all.txt', 'r')

for item in label_file.readlines():
    imag_info = str(item).replace(' ', '').replace('\n', '').split(',')
    imag_newName = 'dataset/' + str(imag_info[0].split('.')[0]) + '#'
    for i in range(8):
        imag_newName += 'class-' + str(i) + '-' + imag_info[i+1] + '_'
    imag_newName += '.jpg'

    os.rename('dataset/' + imag_info[0], imag_newName)
