#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-3-30 下午4:55
# @Author  : ivy_nie
# @File    : test.py
# @Software: PyCharm

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-3-27 下午5:47
# @Author  : ivy_nie
# @File    : test_gbdt.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import  make_classification
from sklearn.metrics import accuracy_score
#加入Graphviz的系统环境变量,为绘制树的结构配置
import os
# os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'

#加入宋体，使得绘制的图能够正常显示中文，避免乱码
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
import sklearn.preprocessing as pre_processin
from sklearn.metrics import mean_squared_error, r2_score

def printlog(info):
    """定义打印格式"""
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "========"*8 + "%s"%nowtime)
    print(info+"...\n\n")

def loadAndPreprocessData():
    # 数据读取及处理
    global data,lgb_train,lgb_vaild,X_vaild,y_vaild,y_train,X_train
    print("step1: reading data...")

    names = ['apptype', 'deviceversion', 'osversion', 'cleanresult_pv', 'cleanmemory_pv', 'coolcpu_pv',
             'one_key_fix_pv', 'authorization_sms_open_pv', 'access_pv', 'avg_access_pv', 'inview_pv', 'avg_inview_pv']
    data = pd.read_table('data/testX1.txt', names=names).head(6000)
    data.dropna()
    # 处理离散特征
    print("data_preprocessing........................................................................................")
    # from sklearn.preprocessing import LabelEncoder
    label = pre_processin.LabelEncoder()
    data['apptype'] = label.fit_transform(data['apptype'])
    data['deviceversion'] = label.fit_transform(data['deviceversion'].astype('str'))
    data['osversion'] = label.fit_transform(data['osversion'].astype('str'))
    # 准备x_train,x_valid,x_test,y_tarin,y_valid,y_test
    x_data = data[
        ['apptype', 'deviceversion', 'osversion', 'cleanresult_pv', 'cleanmemory_pv', 'coolcpu_pv', 'one_key_fix_pv',
         'authorization_sms_open_pv']]
    y_data = data['avg_access_pv']
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    X_train, X_vaild, y_train, y_vaild = train_test_split(x_data, y_data, test_size=0.2, random_state=1024)
    feature_names =  ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']
    categorical_features = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feature_names,
                            categorical_feature=categorical_features)
    lgb_vaild = lgb.Dataset(X_vaild, label=y_vaild, feature_name=feature_names,
                           categorical_feature=feature_names, reference=lgb_train)
def lgb_fun():
    printlog("Step2:setting parameters...")

    boost_round = 5000
    early_stop_rounds = 100
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l1','l2','rmse'],
        'num_leaves': 31,
        'learning_rate': 0.01,
        #'feature_fraction': 0.9,
        #'bagging_fraction': 0.8,
        #'bagging_freq': 5,
        'verbose': 50,
        #'max_depth': 8,
        #'n_estimators': 5000
    }

    printlog("step3:training model...")
    results={}
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=boost_round,
                    valid_sets=(lgb_vaild,lgb_train),
                    early_stopping_rounds=50,
                    evals_result=results)

    printlog('step4:evaluating model...')
    # predict
    y_pred = gbm.predict(X_vaild, num_iteration=gbm.best_iteration)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_vaild, y_pred) ** 0.5)
    # r2_score取值范围在[0,1]之间，越接近1模型越好，反之越差。
    print("R2 score: %.2f" % r2_score(y_vaild, y_pred))

    #计算重要性，并保存
    printlog("Step5:Computer importance...")
    plt.figure(figsize=(12, 6))
    lgb.plot_importance(gbm)
    plt.title("The importance of features")
    plt.savefig('data/importance.png')
    plt.show()

loadAndPreprocessData()
lgb_fun()