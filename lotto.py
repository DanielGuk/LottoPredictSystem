import random as rd
from django.http import HttpResponse
import django

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import csv
import pandas as pd
import os
import io
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '24'

def max_min_num(list):
    i = 0
    num_cnt = []
    for i in range(0, len(list)):
        num_cnt.append(avg_num[i][1])
        i += 1
    num_max = max(num_cnt)
    num_min = min(num_cnt)
    return num_max, num_min


if __name__ == '__main__':
    f = open('lotto.csv', 'r', encoding='utf-8')
    rdr = csv.reader(f)
    # 당첨 숫자 리스트 합치기
    # 평균
    all_num = []
    lst_cnt = []
    tot_cnt = 0
    # 회차 오차 설정
    round_err_range_min = 870.00
    round_err_range_max = 880.00
    for li in rdr:
        i = 0
        tot_cnt += 1
        # 7은 보너스 불포함, 8은 보너스 포함
        for i in range(1, 8):
            all_num.append(li[i])
        lst_cnt.append(li)
    j = 1
    # 1~45 당첨숫자의 평균값(보너스포함)
    avg_num = []
    avg_pct = []
    fig = Figure()
    canvas = FigureCanvas(fig)
    for j in range(1, 46):
        cnt = 0
        for num_cnt in all_num:
            if j == int(num_cnt):
                cnt += 1
        avg_num.append([j, cnt, round(cnt / tot_cnt * 100, 4)])
        avg_pct.append([j, int(cnt / tot_cnt) * 100])


    df_avg = pd.DataFrame(avg_pct)
    df_avg[0].hist()
    # df_avg.to_csv('avg_num_cnt.csv', header = False, index = False)
    # 최대 당첨 횟수 구하기
    max_len = max_min_num(avg_num)[0]
    # 각 숫자의 횟차 구하기
    chk_num_cnt = 0
    chk_round_cnt = []
    Dict_prt = []
    for l in range(1, 10):
        round_cnt = 0
        chk_round = []
        chk_round.append(l)
        for round_cnt in range(round_cnt, len(lst_cnt)):
            m = 0
            for m in range(1, 8):
                if l == int(lst_cnt[round_cnt][m]):
                    chk_round.append(int(lst_cnt[round_cnt][0]))
        sel_Data = chk_round
        sele_Data_idx = 0
        xData_list = []
        yData_list = []

        for sel_Data_idx in range(0, len(sel_Data) - 1):
            sel_Data_idx += 1
            xData_list.append(sel_Data_idx)
            yData_list.append(sel_Data[-1 * sel_Data_idx])
        ## 30개 샘플링 선형 회귀 하여

        xData = xData_list[0:22]
        yData = yData_list[-23:-1]
        W = tf.Variable(tf.random_uniform([1], -100, 100))
        W1 = tf.Variable(tf.random_uniform([1], -100, 100))
        b = tf.Variable(tf.random_uniform([1], -100, 100))

        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)
        H = W * X + b

        cost = tf.reduce_mean(tf.square(H - Y))
        a = tf.Variable(0.001277)
        optimizer = tf.train.GradientDescentOptimizer(a)
        train = optimizer.minimize(cost)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for i in range(5001):
            sess.run(train, feed_dict={X: xData, Y: yData})
            # if i%500 ==0:
            #     print(i, sess.run(cost, feed_dict={X:xData, Y:yData}), sess.run(W), sess.run(b))

        # print(sess.run(H, feed_dict= {X:[31]}))
        pre_dict = []
        pre_dict.append(sess.run(H, feed_dict={X: [23]}))
        # print(int(pre_dict[0]))
        if float(pre_dict[0]) >= round_err_range_min and float(pre_dict[0]) <= round_err_range_max:
            # print(chk_num_cnt, 'ok')
            plt.plot(xData, yData)
            plt.title('LOTTO Predict Round')
            Dict_prt.append('Num :' + str(chk_num_cnt + 1) +'('+ str(avg_num[chk_num_cnt][2]) + '%)'+ ', P_dict Round : ' + str(pre_dict[0]) + ', Latest_Num : ' + str(yData[-4:-1]) )
            plt.legend(Dict_prt)
        chk_num_cnt += 1
        # print(chk_num_cnt, pre_dict[0])
        # print(str(pre_dict[0]))
        print(chk_num_cnt)
    plt.axis([18, 22, 800, 900])
    plt.xlabel('샘플횟수')
    plt.ylabel('당첨회차')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    plt.close(fig)
    # plt.show()
    # response= HttpResponse(buf.getvalue(), content_type='image/png')
    response = HttpResponse(content_type='image/jpg')
    canvas.print_jpg(response)
    f.close()

#123123