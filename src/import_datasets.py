import numpy as np
from scipy.io import loadmat

def get_unimib_data(s="acc"):
    print("Loading UniMiB set ", s)
    X_flat = loadmat("data/UniMiB-SHAR/data/" + s + "_data.mat")[s + "_data"]
    y = loadmat("data/UniMiB-SHAR/data/" + s + "_labels.mat")[s + "_labels"][:,0]
    if(s=="acc"):
        labels = loadmat("data/UniMiB-SHAR/data/" + s + "_names.mat")[s + "_names"][0,:]
    else:
        labels = loadmat("data/UniMiB-SHAR/data/" + s + "_names.mat")[s + "_names"][:,0]
    print("Num instances: ", len(X_flat))
    print("Instance length: ", len(X_flat[0]))

    y = np.array(y - 1)
    X = np.zeros((len(X_flat), 3, 151), dtype='float')
    X[:,0,0:151]=X_flat[:,0:151]
    X[:,1,0:151]=X_flat[:,151:302]
    X[:,2,0:151]=X_flat[:,302:453]
    print(labels)
    return X, y, labels

def get_uci_data():
    print("Loading UCI HAR Dataset")
    X_x = np.genfromtxt("data/UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt" )
    X_y = np.genfromtxt("data/UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt" )
    X_z = np.genfromtxt("data/UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt" )
    X = np.zeros((7352, 3, 128))
    X[:,0,:]=X_x[:,:]
    X[:,1,:]=X_y[:,:]
    X[:,2,:]=X_z[:,:]
    y = np.genfromtxt("data/UCI HAR Dataset/train/y_train.txt")
    y = np.array(y - 1)
    with open("data/UCI HAR Dataset/activity_labels.txt") as f:
        labels = f.read().split('\n')
    labels = labels[:-1]
    return X, y, labels

def get_synthetic_set(num):
    filename = "data/synthetic/synthetic_set{}".format(num)
    #print(filename)
    X = np.genfromtxt(filename + "_data.csv", dtype="float64", delimiter=",")
    #print(X)
    X = np.reshape(X, (len(X), 1, len(X[0])))
    y = np.genfromtxt(filename+"_labels.csv")
    y = np.array(y, dtype="int")
    #print(y)
    labels = []
    for i in range(len(y)):
        labels.append("Class {}".format(i))
    return X, y, labels

if __name__ == "__main__":
    X, y, labels = get_unimib_data("two_classes")
    X, y, labels = get_uci_data()
