#coding:utf8

import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import os

#---------------------------------------------
#       本以外を消すためのマスク生成
#---------------------------------------------
def get_mask(src):
    hsv   = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    ret,  dst = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)
    ret,v_dst = cv2.threshold(v, 70, 255, cv2.THRESH_BINARY)
    dst = cv2.bitwise_and(dst, v_dst)

    return dst

#---------------------------------------------
#           マスクを元に黒抜き画像に
#---------------------------------------------
def convert_black(src):
    bl     = cv2.blur(src ,(10,10) )
    white  = get_mask(bl)
    dst     = cv2.bitwise_and(src,src,mask=white)

    return dst

#---------------------------------------------
#             輪郭抽出
#---------------------------------------------
def get_outline(src):
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray,(10,10))
    ret, mono = cv2.threshold(blur,20,255, cv2.THRESH_BINARY)
    img, contours, hierarchy = cv2.findContours(mono,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    maxContourNum = 0
    for (i,contour) in enumerate(contours) :
        if cv2.contourArea(contour) < cv2.contourArea(contours[i]):
            maxContourNum = i

    maxContour = contours[i]

    maxContourApprox = cv2.approxPolyDP(maxContour, 500, True)

    maxHeight = len(gray[:,0])
    maxWidth  = len(gray[0,:])
    midHeight = maxHeight / 2
    midWidth  = maxWidth  / 2

    leftUp    = [midWidth, midHeight]
    leftDown  = [midWidth, midHeight]
    rightUp   = [midWidth, midHeight]
    rightDown = [midWidth, midHeight]

    # 四隅を見つける
    for point in maxContourApprox:
        #左上
        if leftUp[0] + leftUp[1] > point[0][0] + point[0][1]:
            leftUp[0] = point[0][0]
            leftUp[1] = point[0][1]

        #左下
        if point[0][0] ** 2 + (maxHeight -point[0][1]) **2 < \
            leftDown[0] ** 2 + (maxHeight -leftDown[1]) **2 :
            leftDown[0] = point[0][0]
            leftDown[1] = point[0][1]

        #右上
        if (maxWidth - point[0][0]) ** 2 + point[0][1] **2 < \
            (maxWidth - rightUp[0]) ** 2 + rightUp[1] **2  :
            rightUp[0] = point[0][0]
            rightUp[1] = point[0][1]

        # 右下
        if (maxWidth - point[0][0]) ** 2 + (maxHeight - point[0][1]) **2  < \
            (maxWidth - rightDown[0]) ** 2 + (maxHeight - rightDown[1]) **2  :
            rightDown[0] = point[0][0]
            rightDown[1] = point[0][1]

    #角度補正
    center = (midWidth, midHeight)
    size   = (maxWidth, maxHeight)
    dx = rightDown[0] - rightUp[0]
    dy = rightDown[1] - rightUp[1]
    angle = np.arctan(dy/dx)
    print dx
    print dy
    print angle
    rotation_mat = cv2.getRotationMatrix2D(center, angle , 1.0)
    #src = cv2.warpAffine(src, rotation_mat, size, flags=cv2.INTER_CUBIC)

    #もっかい
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray,(20,20))
    ret, mono = cv2.threshold(blur,100,255, cv2.THRESH_BINARY)
    img, contours, hierarchy = cv2.findContours(mono,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    maxContourNum = 0
    for (i,contour) in enumerate(contours) :
        if cv2.contourArea(contour) < cv2.contourArea(contours[i]):
            maxContourNum = i

    maxContour = contours[i]

    maxContourApprox = cv2.approxPolyDP(maxContour, 500, True)

    # 四隅を見つける
    for point in maxContourApprox:
        #左上
        if leftUp[0] + leftUp[1] > point[0][0] + point[0][1]:
            leftUp[0] = point[0][0]
            leftUp[1] = point[0][1]

        #左下
        if point[0][0] ** 2 + (maxHeight -point[0][1]) **2 < \
            leftDown[0] ** 2 + (maxHeight -leftDown[1]) **2 :
            leftDown[0] = point[0][0]
            leftDown[1] = point[0][1]

        #右上
        if (maxWidth - point[0][0]) ** 2 + point[0][1] **2 < \
            (maxWidth - rightUp[0]) ** 2 + rightUp[1] **2  :
            rightUp[0] = point[0][0]
            rightUp[1] = point[0][1]

        # 右下
        if (maxWidth - point[0][0]) ** 2 + (maxHeight - point[0][1]) **2  < \
            (maxWidth - rightDown[0]) ** 2 + (maxHeight - rightDown[1]) **2  :
            rightDown[0] = point[0][0]
            rightDown[1] = point[0][1]


    detector = cv2.FeatureDetector_create('ORB')


    # ページ区切りを探す
    """
    topMinCount      = maxHeight #最もホワイトピクセルが少ない（半分から上）
    bottomMinCount   = maxHeight #最もホワイトピクセルが少ない（半分から下）

    topMiddlePoint    = None # ページ区切り（上）
    bottomMiddlePoint = None # ページ区切り（下）
    meanX = (leftUp[0] + rightUp[0]) / 2
    meanY = (leftUp[1] + leftDown[1]) / 2
    delta = 300


    for point in maxContour :
        # 中心付近から離れていたらスキップ
        if point[0][0] > meanX + delta or point[0][0] < meanX - delta :
            continue

        topWhitePointCount    = cv2.countNonZero(mono[point[0][1]:meanY , point[0][0]])
        bottomWhitePointCount = cv2.countNonZero(mono[meanY:point[0][1], point[0][0]])

        if topWhitePointCount < topMinCount and point[0][1] < meanY:
            topMinCount = topWhitePointCount
            topMiddlePoint = point[0]
        if bottomWhitePointCount < bottomMinCount and point[0][1] > meanY:
            bottomMinCount = bottomWhitePointCount
            bottomMiddlePoint = point[0]
    """

    cv2.drawContours(src,[maxContour],-1,(0,255,0),3)
    cv2.circle(src,(leftUp[0],leftUp[1]),30,(255,0,0),thickness = 10)
    cv2.circle(src,(leftDown[0],leftDown[1]),30,(255,0,0),thickness = 10)
    cv2.circle(src,(rightUp[0],rightUp[1]),30,(255,0,0),thickness = 10)
    cv2.circle(src,(rightDown[0],rightDown[1]),30,(255,0,0),thickness = 10)

    src[dst>0.01*dst.max()] = [0,0,255]
    #cv2.circle(src,(topMiddlePoint[0],topMiddlePoint[1]),30,(255,0,0),thickness = 10)
    #cv2.circle(src,(bottomMiddlePoint[0],bottomMiddlePoint[1]),30,(255,0,0),thickness = 10)


    #cv2.imshow("test", cv2.resize(src,(800,600)))
    #cv2.imshow("test1", cv2.resize(mono2,(800,600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return src

#---------------------------------------------
#             メイン関数
#---------------------------------------------
# コマンドラインからパス取得
param = sys.argv
path = param[1]

# 出力用のディレクトリ生成
if not os.path.exists(path + "black"):
    os.mkdir(path + "black")

# 指定したディレクトリ下のファイル全てに適用
files = os.listdir(path)
for file in files:

    # 拡張子のないものはスルー
    if len(file.split(".")) == 1:
        continue

    # 処理
    image = cv2.imread(path + file)
    black = convert_black(image)
    image = get_outline(black)

    # 書き込み処理
    print "writing:" + path + "black/" + file
    cv2.imwrite( path + "black/" + file, image)
