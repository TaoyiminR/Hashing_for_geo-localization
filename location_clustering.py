import numpy as np
import scipy.spatial
import scipy.io as sio
from pyproj import Transformer
import scipy.io as sio
import os
from geopy.distance import geodesic

def Cal_Hamming_Dist(B1, B2):
    dist_array = np.zeros([B1.shape[0], B2.shape[0]])
    iter = B1.shape[0]
    q = B2.shape[1]  # max inner product value
    for i in range(iter):
        distH = 0.5 * (q - np.dot(B1[i, :], B2.transpose()))
        dist_array[i, :] = distH
    return dist_array

# shape=N*1*2
def Utm_2_GPS(utmCoordinate):
    trans = Transformer.from_crs("epsg:32649", "epsg:4326")
    GPSCoordinate = np.zeros([utmCoordinate.shape[0], utmCoordinate.shape[1], utmCoordinate.shape[2]], dtype=np.float32)
    for i in range(utmCoordinate.shape[0]):
        for j in range(utmCoordinate.shape[1]):
            GPSCoordinate[i, j, 0], GPSCoordinate[i, j, 1] = trans.transform(utmCoordinate[i, j, 0], utmCoordinate[i, j, 1])
    return GPSCoordinate

def GPS_Cluster(GPSArray, alpha=1., iter=30):
    """
    GPSArray.shape (8884, 5, 2)
    """
    N = GPSArray.shape[1]
    X = np.ones((N, 1), dtype=np.float32) / N
    clusterArray = np.zeros([GPSArray.shape[0], N], dtype=np.float32)

    for imgIndex in range(GPSArray.shape[0]):
        A = alpha*np.exp(- scipy.spatial.distance.cdist(GPSArray[imgIndex, :, :], GPSArray[imgIndex, :, :], metric='euclidean'))
        for i in range(iter):
            for j in range(N):
                X[j][0] = X[j][0]*((np.matmul(A, X)[j][0]) / np.matmul(np.matmul(np.transpose(X), A), X))
        clusterArray[imgIndex] = X[:, 0]
        X[:, 0] = 1/N

    return clusterArray

def Localization(CluArray, GPSArray, Thres=0.1):
    M = CluArray.shape[0]
    N = CluArray.shape[1]
    mask = np.zeros([M, 1], dtype=np.float32)

    for i in range(M):
        for j in range(N):
            if CluArray[i, j] < Thres:
                GPSArray[i, j, :] = 0.
            else:
                mask[i, 0] += 1.

    avgGPSArray = np.sum(GPSArray, axis=1) / mask
    return avgGPSArray

def Localization_Recall(GPSArray, RealLocationArray):
    error = np.power(np.abs(GPSArray - RealLocationArray), 2)
    errorArray = np.sqrt(error[:, 0] + error[:, 1])
    print('errorArray[0]', errorArray[0])

    savePath = './result' + '/' + 'CVACT' + '/'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    sio.savemat(savePath + 'error.mat', {'errorArray': errorArray})

    amount = 0.
    for i in range(errorArray.shape[0]):
        if errorArray[i] <= 0.018:
            amount += 1
    recall = amount / errorArray.shape[0]
    return recall

def Cal_GPS_Distance(GPS1, GPS2):
    distance = geodesic((GPS1[0], GPS1[1]), (GPS2[0], GPS2[1])).m
    return distance

def Img_Localization(topN=5, iter=10, alpha=1., imgInd=1):
    """
    :param topN:
    :param iter:
    :param alpha:
    :param imgIndex:
    :return: avgLocation
    """
    allDataIndex_path = './data/CVACT/ACT_DataIndex.mat'
    allDataIndex = sio.loadmat(allDataIndex_path)
    utm = allDataIndex['utm']
    imgIds = allDataIndex['panoIds']
    trainDataIndex = allDataIndex['trainSet']['trainInd'][0][0] - 1
    testDataIndex = allDataIndex['valSet']['valInd'][0][0] - 1

    realLocationUtm = utm[testDataIndex[:, 0]]
    realLocationGPS = Utm_2_GPS(np.expand_dims(realLocationUtm, axis=1))
    realLocationGPS = np.squeeze(realLocationGPS, axis=1)

    HashDataPath = './result/CVACT/CVACT_hc.mat'
    HashData = sio.loadmat(HashDataPath)
    satBin = HashData['sat_hc']
    grdBin = HashData['grd_hc']
    dist = Cal_Hamming_Dist(satBin, grdBin)

    orderArray = np.argsort(dist[:])
    topNOrderArray = orderArray[:, 0:topN]
    print(topNOrderArray[imgInd])

    topNOrderUtm = np.zeros([topNOrderArray.shape[0], topNOrderArray.shape[1], 2], dtype=np.float32)
    for i in range(topNOrderArray.shape[0]):
        for j in range(topN):
            topNOrderUtm[i, j] = utm[testDataIndex[topNOrderArray[i, j]]]

    topNOrderGPS = Utm_2_GPS(topNOrderUtm)
    print(topNOrderGPS[imgInd])
    clusterArray = GPS_Cluster(topNOrderGPS, alpha, iter)
    print(clusterArray[imgInd])
    avgGPSArray = Localization(clusterArray, topNOrderGPS, Thres=0.1)
    print(avgGPSArray[imgInd])
    recall = Localization_Recall(avgGPSArray, realLocationGPS)
    print('Test set recall：', recall)

def main():
    Img_Localization(topN=5, iter=50, alpha=1., imgInd=1)

if __name__== '__main__':
    main()


