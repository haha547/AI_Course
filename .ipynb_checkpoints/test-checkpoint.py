{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "import numpy as np  \n",
    "import pandas as pd  \n",
    "from keras.utils import np_utils  # 用來後續將 label 標籤轉為 one-hot-encoding  \n",
    "import tensorflow as tf\n",
    "np.random.seed(10)  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.datasets import mnist  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\t[Info] train data= 60,000\n",
      "\t[Info] test  data= 10,000\n"
     ]
    }
   ],
   "source": [
    "(X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()  \n",
    "print(\"\\t[Info] train data={:7,}\".format(len(X_train_image)))  \n",
    "print(\"\\t[Info] test  data={:7,}\".format(len(X_test_image)))  \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\t[Info] Shape of train data=(60000, 28, 28)\n",
      "\t[Info] Shape of train label=(60000,)\n"
     ]
    }
   ],
   "source": [
    "\n",
    "print(\"\\t[Info] Shape of train data=%s\" % (str(X_train_image.shape)))  \n",
    "print(\"\\t[Info] Shape of train label=%s\" % (str(y_train_label.shape))) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt  \n",
    "def plot_image(image):  \n",
    "    fig = plt.gcf()  \n",
    "    fig.set_size_inches(2,2)  \n",
    "    plt.imshow(image, cmap='binary') # cmap='binary' 參數設定以黑白灰階顯示.  \n",
    "    plt.show() "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 144x144 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_image(X_train_image[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_images_labels_predict(images, labels, prediction, idx, num=10):  \n",
    "    fig = plt.gcf()  \n",
    "    fig.set_size_inches(12, 14)  \n",
    "    if num > 25: num = 25  \n",
    "    for i in range(0, num):  \n",
    "        ax=plt.subplot(5,5, 1+i)  \n",
    "        ax.imshow(images[idx], cmap='binary')  \n",
    "        title = \"l=\" + str(labels[idx])  \n",
    "        if len(prediction) > 0:  \n",
    "            title = \"l={},p={}\".format(str(labels[idx]), str(prediction[idx]))  \n",
    "        else:  \n",
    "            title = \"l={}\".format(str(labels[idx]))  \n",
    "        ax.set_title(title, fontsize=10)  \n",
    "        ax.set_xticks([]); ax.set_yticks([])  \n",
    "        idx+=1  \n",
    "    plt.show()  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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vvlG5RNOOEHlol4gg3Cx55579OgR1/633Xab7zP7RUJIvrFjx6psz8hWqlRJ5fPPP993jAceeEDlKlWqRDznzz//rPLbb7+tcn5+vsqe5/mOMWrUKJW7d+8e8Zxhxp1ZAAAAOItmFgAAAM6imQUAAICzysTMbLVq1VSeOnWqyqeccorKN954o8rnnnuuyvYspb0+nEjR68gheVavXq2yPSNrmzdvnu+zjh07JrQmBOO0004LuoSo9uzZo/Jbb72l8owZM3z72DNxNntty6LWIEW42P/cc3NzI27fpUsXlQcPHpzwmhDdrl27VH7yySdVtn//2zOyr732Wtzn3Lp1q8rXXHONyqtWrYq4/9/+9jffZ8OGDYu7jrDiziwAAACcRTMLAAAAZ9HMAgAAwFllYmbW1qRJE5VfeOEFlfv06aOyvf6onffu3es7x/XXX69y/fr14y0TcRgyZIjK9pp6nTp1UtmV+dii1gaM5+dlUWFhYYn2X7t2rcqHDx/2bfPOO++o/OWXX6p84MABlV966aWIx7TXlGzXrp3vnPZalb/88ovK9iw/wseelRw+fHjE7Tt06KByVlaWykcddVRiCkNc7H+/d+7cGXH7iRMnqrxjxw7fNtOmTVPZfq5jw4YNKv/www8q23O65crpe5XXXnut75z280Qu484sAAAAnEUzCwAAAGfRzAIAAMBZZXJm1nbZZZep3LRpU5WHDh2q8uLFi1W+6667fMe034s8cuRIlY899ti468T/vP766yqvWbNGZXt+6NJLL016Tclg/++wc5s2bVJZTuCKel+5/f/JgAEDVLbfkx6NPTNb1FxyhQoVVK5atarKJ554osp9+/ZV+dRTT1XZnumuW7eu75wNGzZUed++fSo3b97ctw+ClZeXp3KPHj3i2v8vf/mLykVdF0i9ihUrqlynTh2V7ZnYjIwMlYuzDr3dM6Snp6u8fft2lWvVqqXyJZdcEvc5XcKdWQAAADiLZhYAAADOopkFAACAs5iZLUKrVq1UnjNnjsoLFixQuXfv3r5jPP300ypv2bJF5UWLFpWgQtjzgva6f/YM01VXXZX0mopj//79Ko8ZMybi9va72cePH5/okkLNfge6iEjjxo1VXr58eYnOcdxxx6ncvXt33zYtWrRQ+YwzzijROW1TpkzxfWbP4dnzlAifBx54QOW0tLS49o+2Di2C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\n",
      "text/plain": [
       "<Figure size 864x1008 with 10 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_images_labels_predict(X_train_image, y_train_label, [], 0, 10)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\t[Info] xTrain: (60000, 784)\n",
      "\t[Info] xTest: (10000, 784)\n"
     ]
    }
   ],
   "source": [
    "x_Train = X_train_image.reshape(60000, 28*28).astype('float32')  \n",
    "x_Test = X_test_image.reshape(10000, 28*28).astype('float32')  \n",
    "print(\"\\t[Info] xTrain: %s\" % (str(x_Train.shape)))  \n",
    "print(\"\\t[Info] xTest: %s\" % (str(x_Test.shape)))  \n",
    "  \n",
    "# Normalization  \n",
    "x_Train_norm = x_Train/255  \n",
    "x_Test_norm = x_Test/255  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\t[Info] Model summary:\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_1 (Dense)              (None, 256)               200960    \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 10)                2570      \n",
      "=================================================================\n",
      "Total params: 203,530\n",
      "Trainable params: 203,530\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from keras.models import Sequential  \n",
    "from keras.layers import Dense  \n",
    "  \n",
    "model = Sequential()  # Build Linear Model  \n",
    "  \n",
    "model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu')) # Add Input/hidden layer  \n",
    "model.add(Dense(units=10, kernel_initializer='normal', activation='softmax')) # Add Hidden/output layer  \n",
    "print(\"\\t[Info] Model summary:\")  \n",
    "model.summary()  \n",
    "print(\"\")  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_TrainOneHot = np_utils.to_categorical(y_train_label) \n",
    "y_TestOneHot = np_utils.to_categorical(y_test_label) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train_label[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]], dtype=float32)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_TrainOneHot[:1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=200, verbose=2)  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import matplotlib.pyplot as plt  \n",
    "def show_train_history(train_history, train, validation):  \n",
    "    plt.plot(train_history.history[train])  \n",
    "    plt.plot(train_history.history[validation])  \n",
    "    plt.title('Train History')  \n",
    "    plt.ylabel(train)  \n",
    "    plt.xlabel('Epoch')  \n",
    "    plt.legend(['train', 'validation'], loc='upper left')  \n",
    "    plt.show()  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'train_history' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-17-866b8c54659d>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mshow_train_history\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrain_history\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'acc'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'val_acc'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m: name 'train_history' is not defined"
     ]
    }
   ],
   "source": [
    "show_train_history(train_history, 'acc', 'val_acc')  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
