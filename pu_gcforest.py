#!/usr/bin/env python
'''
pu_forest.py
Desc :  PU-Learning GCforest algorithm used to predict circRNA-disease associations.
Usage: ./pu_gcforest.py directory-contains-feature-files outDir randomSeed
E.g. : ./pu_gcforest.py ./features ./pu_gcforest-outdir 10
Coder: zhongyue, etc
Created date: 20181225
'''

import argparse
import numpy as np
import sys
from os import listdir
from os.path import isfile, join

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
sys.path.insert(0, "lib")

from gcforest.gcforest import GCForest


if(len(sys.argv) != 4):
    sys.exit("Usage: %s directory-for-input-files outdir randomSeed\n" %(sys.argv[0]))

inpath = sys.argv[1]
outdir = sys.argv[2]
rs = int(sys.argv[3])
nfolds = 5
topN = 500

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", type=str, default=None, help="gcfoest Net Model File")
    args = parser.parse_args()
    return args

def calPRAUC(ranks, nTPs, topN):
    cumPRAUC = 0
    posRecalls = list()
    for i in range(topN):
        if(ranks[i] < nTPs):
            posRecalls.append(1)
        else:
            posRecalls.append(0)

    curSum = posRecalls[0]
    prevRecall = round(posRecalls[0] / nTPs, 4)
    prevPrec = round(posRecalls[0], 4)
    for i in range(1, topN):
        curSum += posRecalls[i]
        recall = round(curSum / nTPs, 4)
        prec   = round(curSum / (i+1), 4)
        cumPRAUC += ((recall - prevRecall) * (prevPrec + prec) / 2)
        prevRecall = recall
        prevPrec = prec

    cumPRAUC = round(cumPRAUC, 4)
    return cumPRAUC

def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 1
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = [
        {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1,
         "max_features": 1},
        {"n_folds": 3, "type": "ExtraTreesClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1,
         "max_features": 1},
        {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1},
        {"n_folds": 3, "type": "RandomForestClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1}
    ]
    config["cascade"] = ca_config
    return config


if __name__ == "__main__":

    dFeatures = [f for f in listdir(inpath) if isfile(join(inpath, f))]
    for df in dFeatures:
        print("Processing %s" % (df))
        dId = df.split('_')[0]
        pf = "/".join((inpath, df))  # processing input file: df

        outfile = ".".join((dId, "txt"))
        of = "/".join((outdir, outfile))
        d = np.loadtxt(pf, delimiter=',')
        p = d[d[:, 24] == 1, :]
        u = d[d[:, 24] == 0, :]
        x_p = p[:, 0:24]
        y_p = p[:, 24]
        x_u = u[:, 0:24]
        X_n = x_u[0]
        y_u = u[:, 24]

        eRecalls = np.zeros(nfolds)
        ePrecisions = np.zeros(nfolds)
        ePRAUCs = np.zeros(nfolds)
        #训练分类器找到可靠负样本
        i = 0
        for i in range(nfolds):
            x_u_u, x_u_m, y_u_u, y_u_m = train_test_split(x_u, y_u, test_size=0.2)
            x = np.concatenate((x_p, x_u_m), axis=0)
            y = np.concatenate((y_p, y_u_m), axis=0)

            scaler = StandardScaler().fit(x)
            x_train_transformed = scaler.transform(x)
            x_u_train_transformed = scaler.transform(x_u_u)

            config = get_toy_config()
            gc1 = GCForest(config)
            gc1.fit_transform(x_train_transformed, y)
            scores = gc1.predict_proba(x_u_train_transformed)[:, 1]
            orderScores = np.argsort(scores)
            orderList = [str(item) for item in orderScores]
            orderStr = ','.join(orderList)
            top = int(y_u.shape[0] * 0.1)
            topNIndex = orderScores[:top]
            t = 0
            while t < top:
                index = topNIndex[t]
                x_n = x_u_u[index]
                X_n = np.vstack((X_n, x_n))
                t += 1
        X_n = X_n[1:, :]
        X_n = np.unique(X_n, axis=0)
        Y_n = np.zeros(X_n.shape[0])

        #利用可靠负样本和已知正样本训练
        ikf = 0
        kf = KFold(n_splits=nfolds, shuffle=True)
        x_p_splits = list(kf.split(x_p))
        x_u_splits = list(kf.split(x_u))
        for ikf in range(nfolds):
            p_train_index, p_test_index = x_p_splits[ikf]
            u_train_index, u_test_index = x_u_splits[ikf]
            x_p_train = x_p[p_train_index]
            y_p_train = y_p[p_train_index]
            x_p_test = x_p[p_test_index]
            y_p_test = y_p[p_test_index]

            x_u_train = x_u[u_train_index]
            y_u_train = y_u[u_train_index]
            x_u_test = x_u[u_test_index]
            y_u_test = y_u[u_test_index]

            x_test = np.concatenate((x_p_test, x_u_test), axis=0)
            y_test = np.concatenate((y_p_test, y_u_test), axis=0)
            x_train = np.concatenate((x_p_train, x_u_train), axis=0)
            y_train = np.concatenate((y_p_train, y_u_train), axis=0)

            X = np.concatenate((x_p_train, X_n), axis=0)
            Y = np.concatenate((y_p_train, Y_n), axis=0)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

            scaler = StandardScaler().fit(X_train)
            X_train_transformed = scaler.transform(X_train)
            X_test_transformed = scaler.transform(X_test)
            x_test_transformed = scaler.transform(x_test)

            config = get_toy_config()
            gc2 = GCForest(config)
            gc2.fit_transform(X_train_transformed, Y_train)
            y_pred = gc2.predict(X_test_transformed)
            acc = accuracy_score(Y_test, y_pred)
            print("Test Accuracy of RandomForest = {:.2f} %".format(acc * 100))

            scores = gc2.predict_proba(x_test_transformed)[:, 1]
            orderScores = np.argsort(-scores)
            orderList = [str(item) for item in orderScores]
            orderStr = ','.join(orderList)
            topNIndex = orderScores[:topN]
            truePosIndex = np.array(range(y_p_test.shape[0]))
            truePosRecall = np.intersect1d(topNIndex, truePosIndex, assume_unique=True)

            recall = truePosRecall.shape[0] / truePosIndex.shape[0]
            precision = truePosRecall.shape[0] / topN
            prauc = calPRAUC(orderScores, y_p_test.shape[0], topN)
            eRecalls[ikf] = recall
            ePrecisions[ikf] = precision
            ePRAUCs[ikf] = prauc
        mRecall = np.mean(eRecalls)
        print(mRecall)
        stdRecall = np.std(eRecalls)
        mPrec = np.mean(ePrecisions)
        stdPrec = np.std(ePrecisions)
        mPRAUC = np.mean(ePRAUCs)
        stdPRAUC = np.std(ePRAUCs)
        recallList = [str(item) for item in eRecalls]
        precList = [str(item) for item in ePrecisions]
        praucList = [str(item) for item in ePRAUCs]
        recallStr = ','.join(recallList)
        precStr = ','.join(precList)
        praucStr = ','.join(praucList)
        with open(of, "a") as output:
            output.write("%s-RandomState%d, mean+-std recall:%.4f,%.4f\n" % (dId, rs, mRecall, stdRecall))
            output.write("%s-RandomState%d, mean+-std precision:%.4f,%.4f\n" % (dId, rs, mPrec, stdPrec))
            output.write("%s-RandomState%d, mean+-std prauc:%.4f,%.4f\n" % (dId, rs, mPRAUC, stdPRAUC))
            output.write("%s-RandomState%d, 5-fold cv recall:%s\n" % (dId, rs, recallStr))
            output.write("%s-RandomState%d, 5-fold cv precision:%s\n" % (dId, rs, precStr))
            output.write("%s-RandomState%d, 5-fold cv prauc:%s\n" % (dId, rs, praucStr))
            output.write("END\n")
