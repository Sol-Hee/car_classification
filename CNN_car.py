gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)

    model=Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding="same",
                     input_shape=(64,64,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="same",
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(516,activation="relu"))
    model.add(Dense(nb_class,activation="softmax"))
    model.summary()

    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    model.fit(X_train,Y_train,batch_size=16,epochs=20,validation_split=0.1)
    score=model.evaluate(X_test,Y_test) # 두가지 결과 : [2.664301872253418,  0.671999990940094] : loss값, accuracy 값
    print('loss: ',score[0], 'accuracy: ',score[1])

  except RuntimeError as e:
        print(e)

import pandas as pd
accuracy=pd.DataFrame(columns=['loss','accuracy'])

for i in range(1,101,10):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)

            model = Sequential()
            model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", input_shape=(64, 64, 3),
                             activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same",
                             activation='relu'))
            model.add(MaxPool2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(516, activation="relu"))
            model.add(Dense(nb_class, activation="softmax"))

            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[
                'accuracy'])
            model.fit(X_train, Y_train, batch_size=32, epochs=i)
        except RuntimeError as e:
            print(e)
        score = model.evaluate(X_test, Y_test)
        accuracy=accuracy.append(pd.Series(score,index=accuracy.columns),ignore_index=True)


accuracy.plot()
accuracy.to_csv("/users/solhee/final project img/bs_32_acc.csv")

# 정규화
def min_max(arg):
    f= arg/(max(arg)-min(arg))
    return f

import random
x=random.randrange(1,100)

# adam 일때 batch , epoch 어느것에 더영향을 받는가
import pandas as pd
accuracy=pd.DataFrame(columns=['loss','accuracy','batch','epoch'])
for i in range(1,101):
    x=random.randrange(1,100)
    y=random.randrange(1,50)
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", input_shape=(64, 64, 3),
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same",
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(516, activation="relu"))
    model.add(Dense(nb_class, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[
        'accuracy'])
    model.fit(X_train, Y_train, batch_size=x, epochs=y)
    score = model.evaluate(X_test, Y_test)
    score.append(x)
    score.append(y)
    accuracy=accuracy.append(pd.Series(score,index=accuracy.columns),ignore_index=True)
score.append(3)

accuracy.to_csv("/users/solhee/final project img/regression.csv",index=False)
accuracy.sort_values(by="accuracy",axis=0,ascending=False)


import matplotlib.pyplot as plt

fig=plt.figure(figsize=(5,5))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(accuracy['batch'],accuracy['epoch'],accuracy['accuracy'],marker='o')
plt.show()

# accuracy 회귀 분석 (캡쳐)
import statsmodels.api as sm
model=sm.OLS(accuracy['accuracy'],accuracy.iloc[:,2:4]).fit()

model.summary()
model.summary()


# drop out
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import matplotlib.pyplot as plt
datagen=ImageDataGenerator(rotation_range=random.randrange(1,180),width_shift_range=0.2,height_shift_range=0.2,
                   shear_range=0.1,zoom_range=0.1,horizontal_flip=True,fill_mode='nearest')

import random
random.randrange(1,180)

import os,glob
caltech_dir="/users/solhee/final project img"
categories=['2015 damas','Bentz','New granduer']
for idx, cat in enumerate(categories):
    image_dir=caltech_dir+"/"+cat # 이미지 세부 폴더 접속
    file=glob.glob(image_dir+"/*.jpg") # 이미지 이름 수집
    for f in file:
        img=load_img(f) # 이미지 로드
        x=img_to_array(img) # array로 변환
        x=x.reshape((1,)+x.shape) # reshape
        i=0
        for batch in datagen.flow(x,save_to_dir=image_dir,
                                       save_prefix='20201108',save_format="jpg"):
            i+=1
            if i>5: # 한사진당 5개로 뻥튀기
                break



from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
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
    label[idx]=1 # 원핫인코딩
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
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
xy=(X_train,X_test,Y_train,Y_test)
np.save("/users/solhee/data/project_image_data2.npy",xy)

X_train,X_test,Y_train,Y_test=np.load("D:/check/project_image_data2.npy",allow_pickle=True)

X_train.shape #  (2508, 64, 64, 3)
Y_train.shape
X_test.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Dropout,Flatten,Dense

score=pd.DataFrame(columns=['loss','accuracy','epochs'])
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)

    model=Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',input_shape=(64,64,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(516,activation="relu"))
    model.add(Dense(nb_class,activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    for i in range(1, 21):
        lst = []
        model.fit(X_train, Y_train, batch_size=32, epochs=1)
        sc = model.evaluate(X_test, Y_test)
        lst.append(sc[0])
        lst.append(sc[1])
        lst.append(i * 10)
        score = score.append(pd.Series(lst, index=score.columns), ignore_index=True)

  except RuntimeError as e:
        print(e)


score.to_csv("/users/solhee/final project img/분열후.csv")

# 가지를 치고 다시 .
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)

    model=Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',input_shape=(64,64,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(516,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(nb_class,activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    import pandas as pd
    score=pd.DataFrame(columns=['loss','accuracy','epochs'])

    for i in range(1,11):
        lst=[]
        model.fit(X_train,Y_train,batch_size=32,epochs=10)
        sc=model.evaluate(X_test,Y_test)
        lst.append(sc[0])
        lst.append(sc[1])
        lst.append(i*10)
        score=score.append(pd.Series(lst,index=score.columns),ignore_index=True)
  except RuntimeError as e:
        print(e)

score.to_csv("/users/solhee/final project img/가지+분열후.csv")
score1=pd.read_csv("/users/solhee/final project img/분열후.csv")
score1=score1.drop('Unnamed: 0',axis=1)
score1['accuracy'].plot()
score['accuracy'].plot()

# 시각화하기
plt.plot(score['epochs'],score['accuracy'],color='red',label='가지치기 후')
# 가지 친것 score
plt.plot(score1['epochs'],score1['accuracy'],color='gray',label='가지치기 전')
plt.legend()
plt.annotate('epochs=60',xy=(60,0.573),xytext=(60,0.58),
             color='blue',
             arrowprops=dict(facecolor='blue',shrink=-3),
             ha='center')


image_w=32
image_h=32
caltech_dir="/users/solhee/final project img"
categories=['2015 damas','Bentz','New granduer']
nb_class=len(categories)
pixels=image_w*image_h*3
X=[]
Y=[]
for idx, value in enumerate(categories):
    label = [0 for i in range(nb_class)]
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
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
xy=(X_train,X_test,Y_train,Y_test)
np.save("/users/solhee/final project img/LeNet-5 32,32.npy",xy)


X_train,X_test,Y_train,Y_test=np.load("D:/check/LeNet-5 32,32.npy",allow_pickle=True)

# LeNet -5
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D
from tensorflow.keras.layers import Dropout,Flatten,Dense

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    model=Sequential()
    model.add(Conv2D(filters=6,kernel_size=(5,5),activation='relu',input_shape=(32,32,3)))
    model.add(AveragePooling2D(pool_size=(2,2))) # 14 x 14

    model.add(Conv2D(filters=16,kernel_size=(5,5),activation='relu')) # 10 x 10
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=120,activation='relu'))
    model.add(Dense(units=84,activation='relu'))
    model.add(Dense(units=3,activation='softmax'))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

    model.fit(X_train,Y_train,batch_size=32,epochs=20)
    model.evaluate(X_test,Y_test)
  except RuntimeError as e:
        print(e)

# 1씩오를 때 마다 Lenet -5 정확도는 ?
score=pd.DataFrame(columns=['loss','accuracy','epochs'])
for i in range(1,21):
    lst=[]
    model.fit(X_train,Y_train,batch_size=32,epochs=1)
    sc=model.evaluate(X_test,Y_test)
    lst.append(sc[0])
    lst.append(sc[1])
    lst.append(i)
    score=score.append(pd.Series(lst,index=score.columns),ignore_index=True)
score['accuracy'].plot()
score.to_csv("/users/solhee/final project img/LeNet-5.csv",index=False)


#LeNet -5 가지치기

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    model=Sequential()
    model.add(Conv2D(filters=6,kernel_size=(5,5),activation='relu',input_shape=(32,32,3)))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=16,kernel_size=(5,5),activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=120,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=84,activation='relu'))
    model.add(Dense(units=3,activation='softmax'))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

    model.fit(X_train,Y_train,batch_size=32,epochs=10)
    model.evaluate(X_test,Y_test)
    # 10 ~ 100까지 정확도 보려고 .
    score=pd.DataFrame(columns=['loss','accuracy','epochs'])

    for i in range(1,11):
        lst=[]
        model.fit(X_train,Y_train,batch_size=32,epochs=10)
        sc=model.evaluate(X_test,Y_test)
        lst.append(sc[0])
        lst.append(sc[1])
        lst.append(100+i*10)
        score=score.append(pd.Series(lst,index=score.columns),ignore_index=True)
  except RuntimeError as e:
        print(e)

score['accuracy'].plot()
plt.plot(score['epochs'],score['accuracy'])
score.to_csv("/users/solhee/final project img/LeNet-5 가지치기.csv",index=False)

# LeNet-5 + 활성화함수 tanh
X_train,X_test,Y_train,Y_test=np.load("/users/solhee/final project img/LeNet-5 32,32 data.npy",allow_pickle=True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    model=Sequential()
    model.add(Conv2D(filters=6,kernel_size=(5,5),activation='tanh',input_shape=(32,32,3)))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=16,kernel_size=(5,5),activation='tanh'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=120,activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(units=84,activation='tanh'))
    model.add(Dense(units=3,activation='softmax'))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    #model.fit(X_train,Y_train,batch_size=32,epochs=10)
    #model.evaluate(X_test,Y_test)

    score=pd.DataFrame(columns=['loss','accuracy','epochs'])

    for i in range(1,11):
        lst=[]
        model.fit(X_train,Y_train,batch_size=32,epochs=10)
        sc=model.evaluate(X_test,Y_test)
        lst.append(sc[0])
        lst.append(sc[1])
        lst.append(i*10)
        score=score.append(pd.Series(lst,index=score.columns),ignore_index=True)
  except RuntimeError as e:
        print(e)

score['accuracy'].plot()
plt.plot(score['epochs'],score['accuracy'])
score.to_csv("/users/solhee/final project img/LeNet-5+가지치기+tanh.csv",index=False)

X_train,Y_train,X_test,Y_test=np.load("/users/solhee/data/project_image_data.npy",allow_pickle=True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    model=Sequential()
    model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding="same",input_shape=(64,64,3),
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="same",
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(516,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(nb_class,activation="softmax"))
    model.summary()

    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
    model.fit(X_train,Y_train,batch_size=32,epochs=60) # 60일 때 최대였으니까.
    score=model.evaluate(X_test,Y_test)


    score=pd.DataFrame(columns=['loss','accuracy','epochs'])

    for i in range(1,21):
        lst=[]
        model.fit(X_train,Y_train,batch_size=32,epochs=10)
        sc=model.evaluate(X_test,Y_test)
        lst.append(sc[0])
        lst.append(sc[1])
        lst.append(i*10)
        score=score.append(pd.Series(lst,index=score.columns),ignore_index=True)
  except RuntimeError as e:
        print(e)
score.to_csv("/users/solhee/final project img/첫 시도 + 가지치기.csv",index=False)
