import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

#classify data


def caeuclDistance(vect1, vect2):
    return np.sqrt(sum(np.power(vect1-vect2,2)))

def initCentroids(dataSet,k):
    numSamples,dim= dataSet.shape
    centroids=np.zeros((k,dim))
    for i in range(k):
        index = int(np.random.uniform(0,numSamples))
        centroids[i,:]=dataSet[index,:]
    return centroids

def kmeans(dataset,k):
    numSamples=dataset.shape[0]
    clusterAssment=np.mat(np.zeros((numSamples,2)))
    clusterChange=True
    centroids=initCentroids(dataset,k)
    while clusterChange:
        clusterChange=False
        for i in range(numSamples):
            minDist=100000.00
            clusterlabel=0
            for j in range(k):
                distance = caeuclDistance(centroids[j,:],dataset[i,:])
                if distance<minDist:
                    minDist=distance
                    clusterlabel=j
            if clusterAssment[i,0]!= clusterlabel:
                clusterChange=True
                clusterAssment[i,:]=clusterlabel,minDist**2
        for j in range(k):
            #why?
            pointsInCluster=dataset[np.nonzero(clusterAssment[:,0].A==j)[0]]
            centroids[j,:]=np.mean(pointsInCluster,axis=0)
    print('cluster complete')
    return centroids, clusterAssment

def datasetdeal():

    li = load_iris()
    flowers = li.data
    labels = li.target

    maxx = np.max(flowers)
    minx = np.min(flowers)
    flowers_bn = (flowers - minx) / (maxx-minx)
    return flowers_bn,labels

def plot(dataSet,k,centroids,clusterAssment):
    numSamples,dim = dataSet.shape
    dimx=0
    dimy=2
    #  0,1   0,2   0,3  1,2   1,3  2,3
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k>len(mark):
        print('Sorry! Your k is too large!')
        return 1
    for i in range(numSamples):
        markindex=int(clusterAssment[i,0])
        plt.plot(dataSet[i,dimx],dataSet[i,dimy],mark[markindex])
    markc=['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i,dimx],centroids[i,dimy],mark[i],markersize=12)
    plt.show()

def timescount(d0,d1):
    num0 = 0
    num1 = 0
    num2 = 0
    for i in range(d0,d1):
        ax = clusterAssment[i]
        bx = np.array(ax)
        cx = int(bx[0][0])
        if cx == 0:
            num0 += 1
        if cx == 1:
            num1 += 1
        if cx == 2:
            num2 += 1
    listq = [num0, num1, num2]
    num=np.max(listq)
    return num

if __name__ == '__main__':
    flowers_bn, labels=datasetdeal()
    k=3
    centroids, clusterAssment = kmeans(flowers_bn, k)
    print(centroids)
    print(clusterAssment.shape)
    num0 = timescount(0, 50)
    num1 = timescount(50, 100)
    num2 = timescount(100, 150)
    #acc计算
    acc=(num0+num1+num2)/150

    print(acc)
    print('show the result!')
    plot(flowers_bn, k, centroids, clusterAssment)