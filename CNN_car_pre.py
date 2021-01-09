from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request as req
from bs4 import BeautifulSoup
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
font_name= font_manager.FontProperties(fname="fonts/NanumBarunpenB.ttf").get_name()
rc('font',family=font_name)
driver= webdriver.Chrome("/data/chromedriver")
url="https://www.google.com/"
driver.get(url)

#구글 검색 입력
element=driver.find_element_by_class_name("gLFyf") # 검색창 찾기
element.send_keys("2020 그랜저") #검색어 입력
element.send_keys("마티즈")
element.send_keys("벤츠 glc 클래스")
element.send_keys('마티즈 "신형"')
element.send_keys('"2015" "다마스"')
element.submit() #클릭
driver.find_element_by_class_name('hide-focus-ring').click() #  a class='q qs'
driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div[300]/a[1]/div[1]/img').click() # 이미지 자세히 클릭

# Exception 처리 : 중간의 관련 검색어 제외
for i in range(5):
    driver.find_element_by_tag_name('body').send_keys(Keys.END)
    time.sleep(5)
error=[]
error_reason=[]
img_url=[]

for i in range(1,501):
    try:
        driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div[{}]/a[1]/div[1]/img'.format(i)).click()
        html = driver.page_source  # 클릭하고 html 얻기
        soup = BeautifulSoup(html, 'html.parser')  # 클릭 완
        time.sleep(3)
        img=soup.select('div.zjoqD > div > div:nth-of-type(2) > a > img')[0].attrs['src']
        img_url.append(img)
    except Exception as err:
        print(err)
        error_reason.append(err)
        error.append(i)
driver.quit()
#soup.select('div.zjoqD > div > div:nth-of-type(2) > a > img')[0].attrs['src'] # 첫번째 이미지 url


# 이미지 url 수집
x=1
img_error=[]
for i,v in enumerate(list(set(img_url))):
    try:
        req.urlretrieve(v,"/users/solhee/final project img/2015 damas 1/"+'damas'+str(x)+".jpg")
        x+=1
    except Exception as err:
        img_error.append(i)


# 스크롤 내리기
for i in range(3):
    driver.find_element_by_tag_name('body').send_keys(Keys.END)
    time.sleep(5)



from PIL import Image
import os, glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dropout,Flatten,Dense

caltech_dir="/users/solhee/final project img"
categories=['2015 damas','Bentz','New granduer']
nb_class=len(categories)
image_w=64
image_h=64
pixels= image_w * image_h * 3

X=[]
Y=[]

for idx, value in enumerate(categories):
    label=[0 for i in range(nb_class)]
    label[idx]=1
    image_dir=caltech_dir+"/"+value
    files=glob.glob(image_dir+"/*.jpg")
    for i, f in enumerate(files):
        img=Image.open(f) # file 오픈
        img=img.convert("RGB") # RGB 로 변환
        img=img.resize((image_w,image_h)) # 이미지 크기 조절
        data=np.asarray(img) # array 변환
        X.append(data) # 데이터넣기
        Y.append(label)


X = np.array(X) # asarray 했어도 다시 array로 정확하게 변환후 담아주기
Y =np.array(Y)

X_train,Y_train,X_test,Y_test=train_test_split(X,Y)
X_train.shape # (250, 64, 64, 3) : 전체 row 수, 모양, 모양, 색상수
X_test.shape
xy=(X_train,X_test,Y_train,Y_test)
np.save("/users/solhee/data/project_image_data.npy",xy)

#X_train,Y_train,X_test,Y_test=np.load("/users/solhee/data/project_image_data.npy",allow_pickle=True)
X_train,Y_train,X_test,Y_test=np.load("D:/check/project_image_data.npy",allow_pickle=True)


categories=['2015 damas','Bentz','New granduer']
nb_class=len(categories)

X_train.shape ## (375, 64, 64, 3)
