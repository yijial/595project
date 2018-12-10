from sklearn import svm, linear_model
from scipy import sparse
from sklearn.model_selection import cross_val_score
import numpy as np
import os
import sys
from sklearn.kernel_approximation import RBFSampler

def main():
    dir = "../toronto_npz/"
    # dir = "../lasvegas_npz/"

    countClass = np.load(dir + "toronto_useful_classes.npy")
    # countClass = np.load(dir + "/lasvegas_useful_classes.npy")
    weight = {}
    for item in countClass:
        if item not in weight:
            weight[item] = 1
        else:
            weight[item] += 1

    countClass = np.unique(countClass)

    rbf_feature = RBFSampler(gamma=0.1)

    for key, value in weight.items():
        if value < 10:
            weight[key] = 1
        else:
            weight[key] = 1 / weight[key]

    print(weight)
    # clf = linear_model.SGDClassifier(loss='hinge', penalty='l2', max_iter=100, 
    #     eta0=0.001, learning_rate='optimal', class_weight=None)

    clf = linear_model.SGDRegressor(penalty='l2', max_iter=100, 
        eta0=0.005, learning_rate='adaptive')

    print(countClass)
    print("Number of unique classes: " + str(len(countClass)))
    i = 0

    for filename in os.listdir(dir):
        if filename.startswith("train"):
            print(filename)
            i += 1
            train = sparse.load_npz(dir + filename).toarray()
            trainLabel = train[:, 1]
            X_Train = train[:, 2:]
            train_Bid = train[:, 0]

            X_Train = rbf_feature.fit_transform(X_Train)

            clf.partial_fit(X_Train, trainLabel)
            # clf.partial_fit(X_Train, trainLabel, classes=countClass)

        
    print(i)

    test = sparse.load_npz(dir + "test.npz")

    test = test.toarray()
    testLabel = test[:, 1]
    X_Test = test[:, 2:]
    test_Bid = test[:, 0]

    X_Test = rbf_feature.fit_transform(X_Test)
    predict = clf.predict(X_Test)

    testLabel = testLabel.reshape((len(testLabel), 1))
    predict = predict.reshape((len(predict), 1))
    test_Bid = test_Bid.reshape((len(test_Bid), 1))

    output = np.concatenate((testLabel, predict, test_Bid), axis=1)
    outputAll = output

    i = 0
    for filename in os.listdir(dir):
        if filename.startswith("test"):
            print("Testing file: " + str(i))
            i += 1
            test = sparse.load_npz(dir + filename).toarray()

            testLabel = test[:, 1]
            X_Test = test[:, 2:]
            test_Bid = test[:, 0]

            X_Test = rbf_feature.fit_transform(X_Test)
            predict = clf.predict(X_Test)

            testLabel = testLabel.reshape((len(testLabel), 1))
            predict = predict.reshape((len(predict), 1))
            test_Bid = test_Bid.reshape((len(test_Bid), 1))

            output = np.concatenate((testLabel, predict, test_Bid), axis=1)
            outputAll = np.concatenate((outputAll, output), axis=0)
        
    np.save(dir + "output.npy", np.array(outputAll))

if __name__ == '__main__':
    main()
