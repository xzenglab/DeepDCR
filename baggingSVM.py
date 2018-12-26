#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
baggingSVM.py
Desc :  Bagging SVM algorithm used to predict circRNA-disease associations.
Usage: ./baggingSVM.py directory-contains-feature-files outDir NumBootstrap randomSeed
E.g. : ./baggingSVM.py ./features ./baggingSVM-outdir 5 1
Coder: linwei, etc
Created date: 20180305
'''
import time
import numpy as np
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
#from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from scipy import stats

import sys
from os import listdir
from os.path import isfile, join

if(len(sys.argv) != 5):
    sys.exit("Usage: %s directory-for-input-files outdir NumBootstrap randomSeed\n" %(sys.argv[0]))

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

inpath = sys.argv[1]
outdir = sys.argv[2]
T = int(sys.argv[3])
rs = int(sys.argv[4])

nfolds = 5
topN = 500
C = np.power(10.0, range(-6, 4))
R = np.power(10.0, range(-2, 4))
nC = len(C)
nR = len(R)
dFeatures = [f for f in listdir(inpath) if isfile(join(inpath, f))]
for df in dFeatures:
    print("Processing %s" %(df) )
    dId = df.split('_')[0]
    pf = "/".join((inpath, df)) #processing input file: df

    outfile = ".".join((dId, "txt"))
    of = "/".join((outdir, outfile))

#    featImp = ".".join((dId, "impFeat")) #feature importance
#    fif = "/".join((outdir, featImp))

    eRecalls = np.zeros(nfolds)
    ePrecisions = np.zeros(nfolds)
    ePRAUCs = np.zeros(nfolds)
    d = np.loadtxt(pf, delimiter = ',')
    p = d[d[:, 24] == 1, :]
    u = d[d[:, 24] == 0, :]
    X_p = p[:, 0:24]
    y_p = p[:, 24]
    X_u = u[:, 0:24]
    y_u = u[:, 24]
        
    #nfolds to evaluate the performance
    ikf = 0
    kf = KFold(n_splits = nfolds, shuffle = True, random_state = rs)
    X_p_splits = list(kf.split(X_p))
    X_u_splits = list(kf.split(X_u))
    for ikf in range(nfolds):
        p_train_index, p_test_index = X_p_splits[ikf]
        u_train_index, u_test_index = X_u_splits[ikf]
        X_p_train = X_p[p_train_index]
        y_p_train = y_p[p_train_index]
        X_p_test  = X_p[p_test_index]
        y_p_test  = y_p[p_test_index]
        
        X_u_train= X_u[u_train_index]
        y_u_train= y_u[u_train_index]
        X_u_test = X_u[u_test_index]
        y_u_test = y_u[u_test_index]
#        print("Train:", X_p_train.shape, "test:", X_p_test.shape)
        start_time = time.time()
        cvMeans = np.zeros( nC * nR )
        cvStds = np.zeros(nC * nR )
        ithPair = 0
        #nested nfolds to select optimal parameters
        kf2 = KFold(n_splits = nfolds, shuffle = True, random_state = rs)
        for c in C:
            for r in R:
                recalls = np.zeros(nfolds) #recall rate per each c-r pair
                X_p_cv_splits = list(kf2.split(X_p_train))
                #X_u_cv_splits = list(kf2.split(X_u_train))
                for ikf2 in range(nfolds):
                    p_train_cv_index, p_val_cv_index = X_p_cv_splits[ikf2]
                    #u_train_cv_index, u_val_cv_index = X_u_cv_splits[ikf2]
                    X_p_cv_train = X_p_train[p_train_cv_index]
                    y_p_cv_train = y_p_train[p_train_cv_index]
                    X_p_cv_val   = X_p_train[p_val_cv_index]
                    y_p_cv_val   = y_p_train[p_val_cv_index]
                
                    #X_u_cv_train = X_u_train[u_train_cv_index]
                    #y_u_cv_train = y_u_train[u_train_cv_index]
                    #X_u_cv_val = X_u_train[u_val_cv_index]
                    #y_u_cv_val = y_u_train[u_val_cv_index]
                    
                    #mix validation + unlabel for transductive learning to see how it perform on validation set
                    X_pu_cv_val = np.concatenate((X_p_cv_val, X_u_train), axis = 0)
                    y_pu_cv_val = np.concatenate((y_p_cv_val,  y_u_train), axis = 0)
                    p_cv_train_size = X_p_cv_train.shape[0]
                    timesClassified = np.zeros(y_pu_cv_val.shape[0])
                    accScores       = np.zeros(y_pu_cv_val.shape[0])
                    for i in range(T):
                        ss = ShuffleSplit(n_splits=1, train_size=p_cv_train_size, test_size=X_pu_cv_val.shape[0] - p_cv_train_size, random_state=i)
                        train_bstrp_index, test_bstrp_index = list(ss.split(X_pu_cv_val))[0]
                        X_pu_cv_train = X_pu_cv_val[train_bstrp_index]
                        X_pu_cv_test = X_pu_cv_val[test_bstrp_index]
                        y_purturb_cv_train = np.ones(p_cv_train_size) * -1 #set the label as neg
                        X_cv = np.concatenate((X_p_cv_train, X_pu_cv_train), axis = 0)
                        y_cv = np.concatenate((y_p_cv_train, y_purturb_cv_train), axis = 0)
                        scaler = StandardScaler().fit(X_cv)
                        X_cv_transformed = scaler.transform(X_cv)
                        X_pu_cv_test_transformed = scaler.transform(X_pu_cv_test)
                        #1. We need the geometric margins, so we need the clf.coef_, and it's possible only when we use the linear kernel
                        #2. Since we don't have many training instances, it makes sense we adopt the linear kernel to avoid overfitting
                        #3. For linear kernel, the liblinear version is much faster than the libsvm version
                        clf = LinearSVC(penalty = "l2", loss = "hinge", C = c, class_weight = { -1: 1, 1: r}, random_state = i)
                        #clf = SVC(C=c, kernel="linear", class_weight = { -1: 1, 1: r})
                        clf.fit(X_cv_transformed, y_cv)
                        #take geometric margin as score!
                        #here, the decision_function() returns functional margin
                        scores = clf.decision_function(X_pu_cv_test_transformed) / LA.norm(clf.coef_)
                        #print("scores.shape:", scores.shape)
                        #next: accumulate the score and count properly, that's why we use ShuffleSplit
                        accScores[test_bstrp_index] += scores
                        timesClassified[test_bstrp_index] += 1
                        #print("Log: finished %d/%d, time elapsed: %.2f" %(i, T, elapsed_time) )
                        nUnclassified = np.sum(timesClassified == 0)
                    avgScores = accScores / timesClassified
                    orderAvgScores = np.argsort(-avgScores) #sort in descent order
                    topNIndex = orderAvgScores[:topN]
                    truePosIndex = np.array(range(y_p_cv_val.shape[0]) ) #they are the firstN rows in the concatenated validation set
                    truePosRecall = np.intersect1d(topNIndex, truePosIndex, assume_unique=True)
                    recall = truePosRecall.shape[0] / truePosIndex.shape[0]
                    #recall = calPRAUC(orderAvgScores, y_p_cv_val.shape[0], topN) #modified@20180424, turn to PRAUC metric to search optimal parameters
                    recalls[ikf2] = recall
                avgRecall = np.mean(recalls)
                cvMeans[ithPair] = avgRecall
                stdRecall = np.std(recalls)
                cvStds[ithPair] = stdRecall
                #print("For cost: %f, class_weight ratio: %f, rank of top %d: average recall: %.2f%%, std of recall: %.2f" %(c, r, topN, avgRecall*100, stdRecall ))
                #print("For each fold:", recalls)
                ithPair += 1
        elapsed_time = time.time() - start_time
        cvMaxMeanIndex = np.argmax(cvMeans)
        optimalC = C[cvMaxMeanIndex // nR]
        optimalR = R[cvMaxMeanIndex % nR]
        #print("cv-MaxMean:", cvMeans[cvMaxMeanIndex], "cv-MaxMean_std:", cvStds[cvMaxMeanIndex], "cvMaxMeanIndex:", cvMaxMeanIndex)
        print("disease:", dId, ", randomSeed:", rs, "ithFold:", ikf, ", optimalC:", optimalC, ", optimalR:", optimalR, ", cv-MaxMean:", cvMeans[cvMaxMeanIndex])
        #print("cross-validation time elapsed: %.2f" %(elapsed_time) )
        #After parameter selection, we evaluate on the test set with the optimal parameters
        X_test = np.concatenate((X_p_test, X_u_test), axis = 0)
        y_test = np.concatenate((y_p_test, y_u_test), axis = 0)
        train_size = X_p_train.shape[0]
        test_size = X_test.shape[0]
        timesClassified = np.zeros(test_size)
        accScores = np.zeros(test_size)
        for i in range(T):
            ss = ShuffleSplit(n_splits=1, train_size=train_size, test_size=X_test.shape[0] - train_size, random_state = i)
            train_bstrp_test_index, test_bstrp_test_index = list(ss.split(X_test))[0]
            X_pu_train = X_test[train_bstrp_test_index]
            X_pu_test  = X_test[test_bstrp_test_index]
            y_purturb_train = np.ones(train_size) * -1
            X_train = np.concatenate((X_p_train, X_pu_train), axis = 0)
            y_train = np.concatenate((y_p_train, y_purturb_train), axis = 0)
            scaler = StandardScaler().fit(X_train)
            X_train_transformed = scaler.transform(X_train)
            X_pu_test_transformed = scaler.transform(X_pu_test)
            clf = LinearSVC(penalty = "l2", loss = "hinge", C = optimalC, class_weight = { -1: 1, 1: optimalR}, random_state = i)
            clf.fit(X_train_transformed, y_train)
#            coefList = [str(item) for item in np.ravel(clf.coef_)]
#            coefStr  = ','.join(coefList)
#            with open(fif, "a") as output:
#                output.write("%s-%dth fold-%dth classifier, feature importance:%s\n" %(dId, ikf, i, coefStr))
            scores = clf.decision_function(X_pu_test_transformed) / LA.norm(clf.coef_)
            accScores[test_bstrp_test_index] += scores
            timesClassified[test_bstrp_test_index] += 1
        avgScores = accScores / timesClassified
        orderAvgScores = np.argsort(-avgScores)
        orderList = [str(item) for item in orderAvgScores]
        orderStr = ','.join(orderList)
        topNIndex = orderAvgScores[:topN]
        truePosIndex = np.array(range(y_p_test.shape[0]) )
        truePosRecall = np.intersect1d(topNIndex, truePosIndex, assume_unique=True)
        recall = truePosRecall.shape[0] / truePosIndex.shape[0]
        precision = truePosRecall.shape[0] / topN
        prauc = calPRAUC(orderAvgScores, y_p_test.shape[0], topN)
        eRecalls[ikf] = recall
        ePrecisions[ikf] = precision
        ePRAUCs[ikf] = prauc
        print("dId: %s, randomState: %d, %dth-fold, recall: %.2f%%, precision: %.2f%%, prauc: %.4f" %(dId, rs, ikf, recall*100, precision*100, prauc))
        with open(of, "a") as output:   
            output.write("%s-T%d-RandomState%d-%dth fold, number of true positive:%d\n" %(dId, T, rs, ikf, y_p_test.shape[0]))
            output.write("%s\n" %(orderStr))
            output.write("END\n")
    mRecall = np.mean(eRecalls)
    stdRecall = np.std(eRecalls)
    mPrec   = np.mean(ePrecisions)
    stdPrec = np.std(ePrecisions)
    mPRAUC = np.mean(ePRAUCs)
    stdPRAUC = np.std(ePRAUCs)
    recallList = [str(item) for item in eRecalls]
    precList   = [str(item) for item in ePrecisions]
    praucList  = [str(item) for item in ePRAUCs]
    recallStr = ','.join(recallList)
    precStr = ','.join(precList)
    praucStr = ','.join(praucList)
    with open (of, "a") as output:
        output.write("%s-T%d-RandomState%d, mean+-std recall:%.4f,%.4f\n" %(dId, T, rs, mRecall, stdRecall))
        output.write("%s-T%d-RandomState%d, mean+-std precision:%.4f,%.4f\n" %(dId, T, rs, mPrec, stdPrec))
        output.write("%s-T%d-RandomState%d, mean+-std prauc:%.4f,%.4f\n" %(dId, T, rs, mPRAUC, stdPRAUC))
        output.write("%s-T%d-RandomState%d, 5-fold cv recall:%s\n" %(dId, T, rs, recallStr))
        output.write("%s-T%d-RandomState%d, 5-fold cv precision:%s\n" %(dId, T, rs, precStr))
        output.write("%s-T%d-RandomState%d, 5-fold cv prauc:%s\n" %(dId, T, rs, praucStr))
        output.write("END\n")    
    print("summary of %s, randomSeed: %d, top %d, mean/std of prauc, mean/std of recall, mean/std of precision: %f,%f,%f,%f,%f,%f" %(dId, rs, topN, mPRAUC, stdPRAUC, mRecall, stdRecall, mPrec, stdPrec))
    print(eRecalls)
    print(ePrecisions)
    print(ePRAUCs)
    print("END")
