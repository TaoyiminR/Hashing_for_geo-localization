
import torchvision.models as models
import torch
import numpy as np
import cv2
import scipy.io as sio
import hdf5storage

def load_model():
    model = models.vgg11()
    pre = torch.load('./ckpt/Pretrain_Nets/vgg11-bbd30ac9.pth')
    model.load_state_dict(pre)
    return model

def load_CVACT_data():
    imgs_path = '../Ours/Data/CVACT/'
    imgsIndex_path = './data/CVACT/ACT_DataIndex.mat'
    allImgs_index = sio.loadmat(imgsIndex_path)

    allImgs_ids = []
    for i in range(0, len(allImgs_index['panoIds'])):
        imgGrd_id = imgs_path + 'streetview/' + allImgs_index['panoIds'][i] + '_grdView.png'
        imgPolarSat_id = imgs_path + 'polarmap/' + allImgs_index['panoIds'][i] + '_satView_polish.png'
        allImgs_ids.append([imgGrd_id, imgPolarSat_id])

    print('training data loading...')
    trainImgs_index = allImgs_index['trainSet']['trainInd'][0][0] - 1
    trainImg_num = len(trainImgs_index)

    imgGrd_train = []
    imgPolarSat_train = []

    for i in range(trainImg_num):
        if i % 100 == 0:
            print(i)
        img = cv2.imread(allImgs_ids[trainImgs_index[i][0]][0])
        if img is None or img.shape[0] != 128 or img.shape[1] != 512:
            print('ground image read fail: %s, ' % (allImgs_ids[trainImgs_index[i][0]][0]))
            continue

        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939 # blue
        img[:, :, 1] -= 116.779 # Green
        img[:, :, 2] -= 123.6 # Red
        imgGrd_train.append(img)

        img = cv2.imread(allImgs_ids[trainImgs_index[i][0]][1])
        if img is None or img.shape[0] != 128 or img.shape[1] != 512:
            print('polar satview image read fail: %s, ' % (allImgs_ids[trainImgs_index[i][0]][1]))
            continue

        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.6
        imgPolarSat_train.append(img)

    print(len(imgGrd_train))
    print('finish training data loading')

    print('test data loading...')
    testImgs_index = allImgs_index['valSet']['valInd'][0][0] - 1
    testImg_num = len(testImgs_index)

    imgGrd_test = []
    imgPolarSat_test = []
    for i in range(testImg_num):
        if i % 100 == 0:
            print(i)
        img = cv2.imread(allImgs_ids[testImgs_index[i][0]][0])
        if img is None or img.shape[0] != 128 or img.shape[1] != 512:
            print('ground image read fail: %s, ' % (allImgs_ids[testImgs_index[i][0]][0]))
            continue

        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939 # blue
        img[:, :, 1] -= 116.779 # Green
        img[:, :, 2] -= 123.6 # Red
        imgGrd_test.append(img)

        img = cv2.imread(allImgs_ids[testImgs_index[i][0]][1])
        if img is None or img.shape[0] != 128 or img.shape[1] != 512:
            print('polar satview image read fail: %s, ' % (allImgs_ids[testImgs_index[i][0]][1]))
            continue

        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.6
        imgPolarSat_test.append(img)

    print('finish test data loading')

    hdf5storage.savemat('./data/CVACT/CVACT_RawData.mat', {'imgGrd_train': imgGrd_train, 'imgPolarSat_train': imgPolarSat_train,
                                                   'imgGrd_test': imgGrd_test, 'imgPolarSat_test': imgPolarSat_test})
    print('file save success!')

def main():
    load_CVACT_data()

if __name__ == '__main__':
    main()