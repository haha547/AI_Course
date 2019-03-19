import numpy as np  
import pandas as pd  
from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding  
np.random.seed(10)
from keras.datasets import mnist 
(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()  
x_Train = X_train_image.reshape(60000, 28*28).astype('float32')  
x_Test = X_test_image.reshape(10000, 28*28).astype('float32')  
print("\t[Info] xTrain: %s" % (str(x_Train.shape)))  
print("\t[Info] xTest: %s" % (str(x_Test.shape)))  
  
# Normalization  
x_Train_norm = x_Train/255  
x_Test_norm = x_Test/255  
y_TrainOneHot = np_utils.to_categorical(y_train_label) 
y_TestOneHot = np_utils.to_categorical(y_test_label) 
y_train_label[0] 
y_TrainOneHot[:1] 

from keras.models import Sequential  
from keras.layers import Dense  
  
model = Sequential()  # Build Linear Model  
  
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu')) # Add Input/hidden layer  
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax')) # Add Hidden/output layer  
print("\t[Info] Model summary:")  
model.summary()  
print("")  
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=200, verbose=2)  

import matplotlib.pyplot as plt  
def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show()  

show_train_history(train_history, 'acc', 'val_acc')  
show_train_history(train_history, 'loss', 'val_loss')  
scores = model.evaluate(x_Test_norm, y_TestOneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  
print("\t[Info] Making prediction to x_Test_norm")  
prediction = model.predict_classes(x_Test_norm)  # Making prediction and save result to prediction  
print()  
print("\t[Info] Show 10 prediction result (From 240):")  
print("%s\n" % (prediction[240:250]))  
  
if isDisplayAvl():  
    plot_images_labels_predict(X_test_image, y_test_label, prediction, idx=240)  
  
print("\t[Info] Error analysis:")  
for i in range(len(prediction)):  
    if prediction[i] != y_test_label[i]:  
        print("\tAt %d'th: %d is with wrong prediction as %d!" % (i, y_test_label[i], prediction[i]))  
print("\t[Info] Display Confusion Matrix:")  
import pandas as pd  
print("%s\n" % pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict']))  
