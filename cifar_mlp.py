from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils import to_categorical
from keras.datasets import cifar10
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

model=Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=10,batch_size=64,validation_data=(x_test,y_test))

loss,accuracy=model.evaluate(x_test,y_test)
print(f'accuracy:{accuracy}')

plt.figure()
plt.plot(history.history['accuracy'],label='Training accuracy')
plt.plot(history.history['val_accuracy'],label='test accuracy')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.show()