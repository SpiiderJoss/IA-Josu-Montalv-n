{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Convolutional Neural Networks"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Importar Librerías"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:12:59.231970Z",
     "start_time": "2018-11-08T00:12:51.800950Z"
    }
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import os\n",
    "import re\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:13:12.550878Z",
     "start_time": "2018-11-08T00:12:59.234748Z"
    }
   },
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
    "import keras\n",
    "from keras.utils import to_categorical\n",
    "from keras.models import Sequential,Input,Model\n",
    "from keras.layers import Dense, Dropout, Flatten\n",
    "from keras.layers import Conv2D, MaxPooling2D\n",
    "from keras.layers.normalization import BatchNormalization\n",
    "from keras.layers.advanced_activations import LeakyReLU"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cargar set de Imágenes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:45.248080Z",
     "start_time": "2018-11-08T00:13:12.553292Z"
    },
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "leyendo imagenes de  /home/likcos/sportimages/\n",
      "/home/likcos/sportimages/futbol 1\n",
      "/home/likcos/sportimages/americano 7617\n",
      "/home/likcos/sportimages/basket 9348\n",
      "Directorios leidos: 3\n",
      "Imagenes en cada directorio [7618, 9348, 8822]\n",
      "suma Total de imagenes en subdirs: 25788\n"
     ]
    }
   ],
   "source": [
    "dirname = os.path.join(os.getcwd(),'/home/likcos/sportimages')\n",
    "imgpath = dirname + os.sep \n",
    "\n",
    "images = []\n",
    "directories = []\n",
    "dircount = []\n",
    "prevRoot=''\n",
    "cant=0\n",
    "\n",
    "print(\"leyendo imagenes de \",imgpath)\n",
    "\n",
    "for root, dirnames, filenames in os.walk(imgpath):\n",
    "    for filename in filenames:\n",
    "        if re.search(\"\\.(jpg|jpeg|png|bmp|tiff)$\", filename):\n",
    "            cant=cant+1\n",
    "            filepath = os.path.join(root, filename)\n",
    "            image = plt.imread(filepath)\n",
    "            images.append(image)\n",
    "            b = \"Leyendo...\" + str(cant)\n",
    "            print (b, end=\"\\r\")\n",
    "            if prevRoot !=root:\n",
    "                print(root, cant)\n",
    "                prevRoot=root\n",
    "                directories.append(root)\n",
    "                dircount.append(cant)\n",
    "                cant=0\n",
    "dircount.append(cant)\n",
    "\n",
    "dircount = dircount[1:]\n",
    "dircount[0]=dircount[0]+1\n",
    "print('Directorios leidos:',len(directories))\n",
    "print(\"Imagenes en cada directorio\", dircount)\n",
    "print('suma Total de imagenes en subdirs:',sum(dircount))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Creamos las etiquetas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:45.269861Z",
     "start_time": "2018-11-08T00:16:45.251786Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cantidad etiquetas creadas:  25788\n"
     ]
    }
   ],
   "source": [
    "labels=[]\n",
    "indice=0\n",
    "for cantidad in dircount:\n",
    "    for i in range(cantidad):\n",
    "        labels.append(indice)\n",
    "    indice=indice+1\n",
    "print(\"Cantidad etiquetas creadas: \",len(labels))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:45.285925Z",
     "start_time": "2018-11-08T00:16:45.273489Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 futbol\n",
      "1 americano\n",
      "2 basket\n"
     ]
    }
   ],
   "source": [
    "deportes=[]\n",
    "indice=0\n",
    "for directorio in directories:\n",
    "    name = directorio.split(os.sep)\n",
    "    print(indice , name[len(name)-1])\n",
    "    deportes.append(name[len(name)-1])\n",
    "    indice=indice+1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:45.498672Z",
     "start_time": "2018-11-08T00:16:45.290061Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total number of outputs :  3\n",
      "Output classes :  [0 1 2]\n"
     ]
    }
   ],
   "source": [
    "y = np.array(labels)\n",
    "X = np.array(images, dtype=np.uint8) #convierto de lista a numpy\n",
    "\n",
    "\n",
    "\n",
    "# Find the unique numbers from the train labels\n",
    "classes = np.unique(y)\n",
    "nClasses = len(classes)\n",
    "print('Total number of outputs : ', nClasses)\n",
    "print('Output classes : ', classes)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Creamos Sets de Entrenamiento y Test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:45.669596Z",
     "start_time": "2018-11-08T00:16:45.502716Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training data shape :  (20630, 21, 28, 3) (20630,)\n",
      "Testing data shape :  (5158, 21, 28, 3) (5158,)\n"
     ]
    }
   ],
   "source": [
    "train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)\n",
    "print('Training data shape : ', train_X.shape, train_Y.shape)\n",
    "print('Testing data shape : ', test_X.shape, test_Y.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:49.319746Z",
     "start_time": "2018-11-08T00:16:45.673944Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Ground Truth : 1')"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=[5,5])\n",
    "\n",
    "# Display the first image in training data\n",
    "plt.subplot(121)\n",
    "plt.imshow(train_X[0,:,:], cmap='gray')\n",
    "plt.title(\"Ground Truth : {}\".format(train_Y[0]))\n",
    "\n",
    "# Display the first image in testing data\n",
    "plt.subplot(122)\n",
    "plt.imshow(test_X[0,:,:], cmap='gray')\n",
    "plt.title(\"Ground Truth : {}\".format(test_Y[0]))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Preprocesamos las imagenes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:50.798162Z",
     "start_time": "2018-11-08T00:16:49.322999Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x7f32d6b796d8>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "train_X = train_X.astype('float32')\n",
    "test_X = test_X.astype('float32')\n",
    "train_X = train_X/255.\n",
    "test_X = test_X/255.\n",
    "plt.imshow(test_X[0,:,:])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Hacemos el One-hot Encoding para la red"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:50.815482Z",
     "start_time": "2018-11-08T00:16:50.800831Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original label: 1\n",
      "After conversion to one-hot: [0. 1. 0.]\n"
     ]
    }
   ],
   "source": [
    "# Change the labels from categorical to one-hot encoding\n",
    "train_Y_one_hot = to_categorical(train_Y)\n",
    "test_Y_one_hot = to_categorical(test_Y)\n",
    "\n",
    "# Display the change for category label using one-hot encoding\n",
    "print('Original label:', train_Y[0])\n",
    "print('After conversion to one-hot:', train_Y_one_hot[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Creamos el Set de Entrenamiento y Validación"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:51.218520Z",
     "start_time": "2018-11-08T00:16:50.818992Z"
    }
   },
   "outputs": [],
   "source": [
    "#Mezclar todo y crear los grupos de entrenamiento y testing\n",
    "train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:51.228116Z",
     "start_time": "2018-11-08T00:16:51.222460Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(16504, 21, 28, 3) (4126, 21, 28, 3) (16504, 3) (4126, 3)\n"
     ]
    }
   ],
   "source": [
    "print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Creamos el modelo de CNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:51.244776Z",
     "start_time": "2018-11-08T00:16:51.238704Z"
    }
   },
   "outputs": [],
   "source": [
    "#declaramos variables con los parámetros de configuración de la red\n",
    "INIT_LR = 1e-3 # Valor inicial de learning rate. El valor 1e-3 corresponde con 0.001\n",
    "epochs = 20 # Cantidad de iteraciones completas al conjunto de imagenes de entrenamiento\n",
    "batch_size = 64 # cantidad de imágenes que se toman a la vez en memoria"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:51.384131Z",
     "start_time": "2018-11-08T00:16:51.252188Z"
    }
   },
   "outputs": [],
   "source": [
    "sport_model = Sequential()\n",
    "sport_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(21,28,3)))\n",
    "sport_model.add(LeakyReLU(alpha=0.1))\n",
    "sport_model.add(MaxPooling2D((2, 2),padding='same'))\n",
    "sport_model.add(Dropout(0.5))\n",
    "\n",
    "sport_model.add(Flatten())\n",
    "sport_model.add(Dense(32, activation='linear'))\n",
    "sport_model.add(LeakyReLU(alpha=0.1))\n",
    "sport_model.add(Dropout(0.5))\n",
    "sport_model.add(Dense(nClasses, activation='softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:51.401674Z",
     "start_time": "2018-11-08T00:16:51.386676Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_3\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_3 (Conv2D)            (None, 21, 28, 32)        896       \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_5 (LeakyReLU)    (None, 21, 28, 32)        0         \n",
      "_________________________________________________________________\n",
      "max_pooling2d_3 (MaxPooling2 (None, 11, 14, 32)        0         \n",
      "_________________________________________________________________\n",
      "dropout_5 (Dropout)          (None, 11, 14, 32)        0         \n",
      "_________________________________________________________________\n",
      "flatten_3 (Flatten)          (None, 4928)              0         \n",
      "_________________________________________________________________\n",
      "dense_5 (Dense)              (None, 32)                157728    \n",
      "_________________________________________________________________\n",
      "leaky_re_lu_6 (LeakyReLU)    (None, 32)                0         \n",
      "_________________________________________________________________\n",
      "dropout_6 (Dropout)          (None, 32)                0         \n",
      "_________________________________________________________________\n",
      "dense_6 (Dense)              (None, 3)                 99        \n",
      "=================================================================\n",
      "Total params: 158,723\n",
      "Trainable params: 158,723\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "sport_model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:16:51.472349Z",
     "start_time": "2018-11-08T00:16:51.406817Z"
    }
   },
   "outputs": [],
   "source": [
    "sport_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adagrad(lr=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Entrenamos el modelo: Aprende a clasificar imágenes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:20:49.562522Z",
     "start_time": "2018-11-08T00:16:51.474807Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 16504 samples, validate on 4126 samples\n",
      "Epoch 1/20\n",
      "16504/16504 [==============================] - 6s 368us/step - loss: 0.5977 - accuracy: 0.7813 - val_loss: 0.4250 - val_accuracy: 0.8808\n",
      "Epoch 2/20\n",
      "16504/16504 [==============================] - 6s 355us/step - loss: 0.4525 - accuracy: 0.8523 - val_loss: 0.3706 - val_accuracy: 0.8905\n",
      "Epoch 3/20\n",
      "16504/16504 [==============================] - 6s 355us/step - loss: 0.4106 - accuracy: 0.8680 - val_loss: 0.3456 - val_accuracy: 0.8982\n",
      "Epoch 4/20\n",
      "16504/16504 [==============================] - 6s 360us/step - loss: 0.3832 - accuracy: 0.8768 - val_loss: 0.3254 - val_accuracy: 0.9028\n",
      "Epoch 5/20\n",
      "16504/16504 [==============================] - 7s 452us/step - loss: 0.3643 - accuracy: 0.8860 - val_loss: 0.3079 - val_accuracy: 0.9033\n",
      "Epoch 6/20\n",
      "16504/16504 [==============================] - 6s 371us/step - loss: 0.3449 - accuracy: 0.8906 - val_loss: 0.2908 - val_accuracy: 0.9074\n",
      "Epoch 7/20\n",
      "16504/16504 [==============================] - 7s 432us/step - loss: 0.3332 - accuracy: 0.8955 - val_loss: 0.2821 - val_accuracy: 0.9106\n",
      "Epoch 8/20\n",
      "16504/16504 [==============================] - 7s 411us/step - loss: 0.3223 - accuracy: 0.8961 - val_loss: 0.2705 - val_accuracy: 0.9120\n",
      "Epoch 9/20\n",
      "16504/16504 [==============================] - 8s 459us/step - loss: 0.3146 - accuracy: 0.9002 - val_loss: 0.2629 - val_accuracy: 0.9137\n",
      "Epoch 10/20\n",
      "16504/16504 [==============================] - 7s 427us/step - loss: 0.3030 - accuracy: 0.9038 - val_loss: 0.2565 - val_accuracy: 0.9142\n",
      "Epoch 11/20\n",
      "16504/16504 [==============================] - 7s 407us/step - loss: 0.2997 - accuracy: 0.9042 - val_loss: 0.2522 - val_accuracy: 0.9142\n",
      "Epoch 12/20\n",
      "16504/16504 [==============================] - 8s 458us/step - loss: 0.2897 - accuracy: 0.9085 - val_loss: 0.2465 - val_accuracy: 0.9164\n",
      "Epoch 13/20\n",
      "16504/16504 [==============================] - 7s 427us/step - loss: 0.2834 - accuracy: 0.9106 - val_loss: 0.2396 - val_accuracy: 0.9178\n",
      "Epoch 14/20\n",
      "16504/16504 [==============================] - 8s 465us/step - loss: 0.2784 - accuracy: 0.9108 - val_loss: 0.2347 - val_accuracy: 0.9190\n",
      "Epoch 15/20\n",
      "16504/16504 [==============================] - 8s 460us/step - loss: 0.2683 - accuracy: 0.9153 - val_loss: 0.2291 - val_accuracy: 0.9215\n",
      "Epoch 16/20\n",
      "16504/16504 [==============================] - 7s 420us/step - loss: 0.2702 - accuracy: 0.9111 - val_loss: 0.2263 - val_accuracy: 0.9224\n",
      "Epoch 17/20\n",
      "16504/16504 [==============================] - 7s 417us/step - loss: 0.2629 - accuracy: 0.9175 - val_loss: 0.2223 - val_accuracy: 0.9232\n",
      "Epoch 18/20\n",
      "16504/16504 [==============================] - 7s 440us/step - loss: 0.2650 - accuracy: 0.9151 - val_loss: 0.2198 - val_accuracy: 0.9246\n",
      "Epoch 19/20\n",
      "16504/16504 [==============================] - 7s 441us/step - loss: 0.2564 - accuracy: 0.9170 - val_loss: 0.2153 - val_accuracy: 0.9254\n",
      "Epoch 20/20\n",
      "16504/16504 [==============================] - 7s 423us/step - loss: 0.2513 - accuracy: 0.9182 - val_loss: 0.2115 - val_accuracy: 0.9246\n"
     ]
    }
   ],
   "source": [
    "# este paso puede tomar varios minutos, dependiendo de tu ordenador, cpu y memoria ram libre\n",
    "sport_train = sport_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:20:49.676566Z",
     "start_time": "2018-11-08T00:20:49.566203Z"
    }
   },
   "outputs": [],
   "source": [
    "# guardamos la red, para reutilizarla en el futuro, sin tener que volver a entrenar\n",
    "sport_model.save(\"sports_mnist.h5py\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Evaluamos la red"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:20:54.462929Z",
     "start_time": "2018-11-08T00:20:49.678643Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5158/5158 [==============================] - 1s 116us/step\n"
     ]
    }
   ],
   "source": [
    "test_eval = sport_model.evaluate(test_X, test_Y_one_hot, verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:20:54.474683Z",
     "start_time": "2018-11-08T00:20:54.465378Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test loss: 0.20198906165955371\n",
      "Test accuracy: 0.931562602519989\n"
     ]
    }
   ],
   "source": [
    "print('Test loss:', test_eval[0])\n",
    "print('Test accuracy:', test_eval[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'val_loss': [0.42503637669159267,\n",
       "  0.3705766632380095,\n",
       "  0.34558884389933253,\n",
       "  0.32540337395390667,\n",
       "  0.30794071532890627,\n",
       "  0.29076717693541043,\n",
       "  0.2821179268664084,\n",
       "  0.2704893363747155,\n",
       "  0.26286337668030113,\n",
       "  0.25653135595943416,\n",
       "  0.25224101396076204,\n",
       "  0.2464507342078301,\n",
       "  0.2396305939755969,\n",
       "  0.23467521045142103,\n",
       "  0.22914478338810637,\n",
       "  0.22631870211615357,\n",
       "  0.22232820383276225,\n",
       "  0.2197963333121215,\n",
       "  0.21529969848219307,\n",
       "  0.21153392847510027],\n",
       " 'val_accuracy': [0.8807561993598938,\n",
       "  0.8904507756233215,\n",
       "  0.8982064723968506,\n",
       "  0.9028114676475525,\n",
       "  0.903296172618866,\n",
       "  0.9074164032936096,\n",
       "  0.9105671644210815,\n",
       "  0.9120213389396667,\n",
       "  0.9137178659439087,\n",
       "  0.9142026305198669,\n",
       "  0.9142026305198669,\n",
       "  0.9163839221000671,\n",
       "  0.9178380966186523,\n",
       "  0.9190499186515808,\n",
       "  0.9214735627174377,\n",
       "  0.9224430322647095,\n",
       "  0.9231701493263245,\n",
       "  0.9246243238449097,\n",
       "  0.9253514409065247,\n",
       "  0.9246243238449097],\n",
       " 'loss': [0.5977478268160383,\n",
       "  0.4524504510424869,\n",
       "  0.4106068651594124,\n",
       "  0.3832451606253311,\n",
       "  0.3642802497483086,\n",
       "  0.34487915885604475,\n",
       "  0.3332306780516957,\n",
       "  0.3223062798749228,\n",
       "  0.31456335239253225,\n",
       "  0.3029654643602644,\n",
       "  0.2996689026296803,\n",
       "  0.28969534331597696,\n",
       "  0.28336069484792514,\n",
       "  0.27835490759344744,\n",
       "  0.26831972648146313,\n",
       "  0.2701836408224721,\n",
       "  0.2629434983821616,\n",
       "  0.2649775531126698,\n",
       "  0.2563717053951249,\n",
       "  0.2513469180116593],\n",
       " 'accuracy': [0.78132576,\n",
       "  0.85233885,\n",
       "  0.868032,\n",
       "  0.87681776,\n",
       "  0.885967,\n",
       "  0.890572,\n",
       "  0.8955405,\n",
       "  0.8960858,\n",
       "  0.900206,\n",
       "  0.9037809,\n",
       "  0.904205,\n",
       "  0.90850705,\n",
       "  0.9106277,\n",
       "  0.9108095,\n",
       "  0.9152933,\n",
       "  0.91105187,\n",
       "  0.9175351,\n",
       "  0.9151115,\n",
       "  0.9169898,\n",
       "  0.9182016]}"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sport_train.history"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:20:55.014647Z",
     "start_time": "2018-11-08T00:20:54.479693Z"
    },
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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+fl7dQJk27ZtDBkyhKlTp3LNNdfwwAMPMGnSpH0e29159dVXmTt3LpMnT+bvf/87d911F4cccgh/+ctfeOONN+jfv3+l8RUWFnLjjTdSUFBA69atOeWUU/jb3/5Gbm4umzdvZunSpQBs3boVgNtuu413332Xxo0bl26rC0nV0bv7M+5+tLsf6e5Tom03RUkeD65x93x37+3us+OOfcDdj4qWqn9f7afx42HdOtizJ9wqyYukXl23hx155JGlSR7gkUceoX///vTv358VK1awfPnyfY5p1qwZp59+OgDHHXcc69atS/jY3/zmN/cp8/LLLzN27FgAjj32WHr27FlpfAsWLOCkk06iQ4cONGrUiHPPPZf58+dz1FFHsXLlSq666irmzZtH69atAejZsyfnnXceM2fOrNYFTzWlIRBEJGkVtXvVVntYixYtSu+vWrWKO+64gxdeeIElS5YwYsSIhP3JGzduXHo/JyeH4uLihI/dpEmTKsvsr/bt27NkyRIGDx7MjBkzuOyyywCYN28el19+OQsXLmTAgAHs3r07pc9bESV6EUlaOtvDPvnkE1q1asVBBx3Exo0bmTcv5c19DBo0iDlz5gCwdOnShL8Y4h1//PG8+OKLbNmyheLiYmbPns2QIUMoKirC3fnWt77F5MmTee2119i9ezeFhYWcdNJJ3HbbbWzevJmd5evBaonGuhGRpKWzPax///7k5+fTvXt3Dj/8cAYNGpTy5/jv//5vzj//fPLz80uXkmqXRDp16sQtt9zC0KFDcXfOOusszjzzTF577TUuvvhi3B0z49Zbb6W4uJhzzz2X7du3s2fPHq699lpatWqV8teQSMbNGRuLxVwTj4jUnRUrVtCjR490h5ERiouLKS4upmnTpqxatYrhw4ezatUqGjbMrHPiRH8zM1vk7rFE5TMrehGRNNqxYwcnn3wyxcXFuDv33HNPxiX5/VH/X4GISIq0adOGRYsWpTuMlFNjrIhIllOiFxHJckr0IiJZToleRCTLKdGLSFoNGzZsn4ufpk+fzhVXXFHpcS1btgRgw4YNjBkzJmGZoUOHUlV37enTp+914dIZZ5yRknFobr75Zm6//fYaP04qKNGLSFqNGzeO2bNn77Vt9uzZjBs3LqnjDzvsMB577LH9fv7yif6ZZ56hTZukp9OoF5ToRSStxowZw9NPP106yci6devYsGEDgwcPLu3X3r9/f3r37s2TTz65z/Hr1q2jV69eAHz22WeMHTuWHj16MHr0aD777LPScldccUXpEMc/+clPALjzzjvZsGEDw4YNY9iwYQDk5eWxefNmAKZNm0avXr3o1atX6RDH69ato0ePHlx66aX07NmT4cOH7/U8iSxevJiBAwfSp08fRo8ezccff1z6/CXDFpcMpvbSSy+VTrzSr18/tm/fvt/vbQn1oxeRUldfDameOKlvX4hyZELt2rVjwIABPPvss4waNYrZs2dzzjnnYGY0bdqUxx9/nIMOOojNmzczcOBARo4cWeG8qb/73e9o3rw5K1asYMmSJXsNMzxlyhTatWvH7t27Ofnkk1myZAlXXXUV06ZN48UXX6RDhw57PdaiRYt48MEHWbBgAe7O8ccfz5AhQ2jbti2rVq3ikUce4b777uOcc87hL3/5S6Xjy59//vncddddDBkyhJtuuomf/vSnTJ8+nalTp/LOO+/QpEmT0uqi22+/nRkzZjBo0CB27NhB06ZNq/FuJ6YzehFJu/jqm/hqG3fn+uuvp0+fPpxyyimsX7+eTZs2Vfg48+fPL024ffr0oU+fPqX75syZQ//+/enXrx/Lli2rcsCyl19+mdGjR9OiRQtatmzJN7/5Tf79738D0LVrV/r27QtUPhQyhPHxt27dypAhQwC44IILmD9/fmmM48eP5+GHHy69AnfQoEFcc8013HnnnWzdujUlV+bqjF5ESlV25l2bRo0axfe//31ee+01du7cyXHHHQfAzJkzKSoqYtGiRTRq1Ii8vLyEQxNX5Z133uH2229n4cKFtG3blgsvvHC/HqdEyRDHEIY5rqrqpiJPP/008+fP56mnnmLKlCksXbqUSZMmceaZZ/LMM88waNAg5s2bR/fu3fc7VtAZvYhkgJYtWzJs2DC++93v7tUIu23bNr7yla/QqFEjXnzxRd5NNDl0nBNPPJFZs2YB8Oabb7JkyRIgDHHcokULWrduzaZNm3j22WdLj2nVqlXCevDBgwfzxBNPsHPnTj799FMef/xxBg8eXO3X1rp1a9q2bVv6a+BPf/oTQ4YMYc+ePbz//vsMGzaMW2+9lW3btrFjxw7WrFlD7969+dGPfsRXv/pV3nrrrWo/Z3lJndGb2QjgDsJUgve7+9Ry+y8EfknZpOG/cff7o327gaXR9vfcfWSNoxaRrDNu3DhGjx69Vw+c8ePHc9ZZZ9G7d29isViVZ7ZXXHEFF110ET169KBHjx6lvwyOPfZY+vXrR/fu3encufNeQxxPmDCBESNGcNhhh/Hiiy+Wbu/fvz8XXnghAwYMAOCSSy6hX79+lVbTVOShhx7i8ssvZ+fOnRxxxBE8+OCD7N69m/POO49t27bh7lx11VW0adOGH//4x7z44os0aNCAnj17ls6WVRNVDlNsZjnA28CpQCFh/tdx7r48rsyFQMzdJyY4foe7t0w2IA1TLFK3NExx/VPdYYqTqboZAKx297XuvguYDYyqcaQiIlInkkn0HYH349YLo23lnW1mS8zsMTPrHLe9qZkVmNkrZvaNRE9gZhOiMgVFRUXJRy8iIlVKVWPsU0Ceu/cBngceitt3ePRz4lxgupkdWf5gd7/X3WPuHsvNzU1RSCKSrEybaU4qtj9/q2QS/Xog/gy9E2WNriVPvMXdv4hW7weOi9u3PrpdC/wL6FftKEWk1jRt2pQtW7Yo2dcD7s6WLVuqfRFVMr1uFgLdzKwrIcGPJZydlzKzQ919Y7Q6ElgRbW8L7HT3L8ysAzAIuK1aEYpIrerUqROFhYWo2rR+aNq0KZ06darWMVUmencvNrOJwDxC98oH3H2ZmU0GCtx9LnCVmY0EioGPgAujw3sA95jZHsKvh6nxvXVEJP0aNWpE165d0x2G1KIqu1fWNXWvFBGpvpp2rxQRkXpMiV5EJMsp0YuIZDklehGRLKdELyKS5ZToRUSynBK9iEiWU6IXEclySvQiIllOiV5EJMsp0YuIZDklehGRLKdELyKS5ZToRUSynBK9iEiWU6IXEclySSV6MxthZivNbLWZTUqw/0IzKzKzxdFySdy+C8xsVbRckMrgRUSkalVOJWhmOcAM4FSgEFhoZnMTTAn4qLtPLHdsO+AnQAxwYFF07McpiV5ERKqUzBn9AGC1u691913AbGBUko9/GvC8u38UJffngRH7F6qIiOyPZBJ9R+D9uPXCaFt5Z5vZEjN7zMw6V+dYM5tgZgVmVqCZ6EVEUitVjbFPAXnu3odw1v5QdQ5293vdPebusdzc3BSFJCIikFyiXw90jlvvFG0r5e5b3P2LaPV+4LhkjxURkdqVTKJfCHQzs65m1hgYC8yNL2Bmh8atjgRWRPfnAcPNrK2ZtQWGR9tERKSOVNnrxt2LzWwiIUHnAA+4+zIzmwwUuPtc4CozGwkUAx8BF0bHfmRmtxC+LAAmu/tHtfA6RESkAubu6Y5hL7FYzAsKCtIdhohIvWJmi9w9lmifrowVEclySvQiIllOiV5EJMsp0YuIZDklehGRLKdELyKS5ZToRUSynBK9iEiWU6IXEclySvQiIllOiV5EJMsp0YuIZDklehGRLKdELyKS5ZToRUSynBK9iEiWU6KPzJwJeXnQoEG4nTkz3RGJiKRGUonezEaY2UozW21mkyopd7aZuZnFovU8M/vMzBZHy92pCjyVZs6ECRPg3XfBPdxOmKBkLyLZocpEb2Y5wAzgdCAfGGdm+QnKtQK+Bywot2uNu/eNlstTEHPK3XAD7Ny597adO8N2EZH6Lpkz+gHAandf6+67gNnAqATlbgFuBT5PYXx14r33qrddRKQ+SSbRdwTej1svjLaVMrP+QGd3fzrB8V3N7HUze8nMBid6AjObYGYFZlZQVFSUbOwp06VL9baLiNQnNW6MNbMGwDTgBwl2bwS6uHs/4BpglpkdVL6Qu9/r7jF3j+Xm5tY0pGqbMgWaN997W/PmYbuISH2XTKJfD3SOW+8UbSvRCugF/MvM1gEDgblmFnP3L9x9C4C7LwLWAEenIvBUGj8e7r0XDj8czMLtvfeG7SIi9V3DJMosBLqZWVdCgh8LnFuy0923AR1K1s3sX8C17l5gZrnAR+6+28yOALoBa1MYf8qMH6/ELiLZqcpE7+7FZjYRmAfkAA+4+zIzmwwUuPvcSg4/EZhsZl8Ce4DL3f2jVAQuIiLJMXdPdwx7icViXlBQkO4wRETqFTNb5O6xRPt0ZayISJZTohcRyXJK9CIiWU6JXkQkyynRi4hkuaxK9G+/DV98ke4oREQyS9Yk+rffhp494a670h2JiEhmyZpEf/TRMHw43HILfPhhuqMREckcWZPoAaZNC+PI33hjuiMREckcWZXojzkGJk6E+++HxYvTHY2ISGbIqkQPcNNN0L49XH11mBZQRORAl3WJvm3bUE//0kvw17/W3fNqcnERyVRZl+gBLrkEeveGa6+Fz+tgYkNNLi4imSwrE33DhjB9OqxbFxpoa5smFxeRTJaViR7gpJPgG9+An/8cNmyo3efS5OIiksmyNtED3H47fPklXH997T6PJhcXkUyWVKI3sxFmttLMVpvZpErKnW1mbmaxuG3XRcetNLPTUhF0so48MvS+eeghWLiw9p5Hk4uLSCarMtGbWQ4wAzgdyAfGmVl+gnKtgO8BC+K25RPmmO0JjAB+Gz1enbnhBjj44NrtbqnJxUUkkyVzRj8AWO3ua919FzAbGJWg3C3ArUB8P5dRwGx3/8Ld3wFWR49XZw46KJxZ/+c/MHt27T3P+PGh8XfPnnCrJC8imSKZRN8ReD9uvTDaVsrM+gOd3f3p6h4bHT/BzArMrKCoqCipwKvjwguhf3/44Q/37R0jIpLtatwYa2YNgGnAD/b3Mdz9XnePuXssNze3piHtIycndLcsLIRf/jLlDy8iktGSSfTrgc5x652ibSVaAb2Af5nZOmAgMDdqkK3q2DozeDCccw7ceiu8/37V5UVEskUyiX4h0M3MuppZY0Lj6tySne6+zd07uHueu+cBrwAj3b0gKjfWzJqYWVegG/Bqyl9Fkm67LTTITqqw35CISPapMtG7ezEwEZgHrADmuPsyM5tsZiOrOHYZMAdYDvwduNLdd9c87P1z+OFhWIRZs0LjrIjIgcA8w4Z4jMViXlBQUGuPv2NHGM74sMNgwYIwCJmISH1nZovcPZZo3wGX5lq2hKlToaAA/vSndEdTRqNfikhtOeASPYQ+7gMGwHXXhTP8dNPolyJSmw7IRN+gAdxxB2zcCL/4Rbqj0eiXIlK7DshEDzBwIJx3HvzqV/DOO+mNRaNfikhtOmATPYS6+pyccMVsOmn0SxGpTQd0ou/YMfSpf+yxMPVgumj0SxGpTQd0oofQr75LlzC65e409fDX6JciUpsO+ETfrFm4YnbxYnjwwfTFodEvRaS2HPCJHsIYOCecEGai2rYt3dGIiKSWEj2humT6dNi8GU4/PXS7FBHJFkr0keOOgzlzYMmScL++jYWjK2tFpCJK9HHGjIFXXgk9XoYODQ2i9YGurBWRyijRl9OrV5hI/OST4bLLQsL84ot0R1U5XVkrIpVRok+gbVv4299C4+x994Wz+w0b0h1VxXRlrYhURom+Ajk54YKlxx6DpUtDvf3//m+6o0pMV9aKSGWU6Ktw9tlh3PqWLWHYMLj77lAPnkl0Za2IVEaJPgk9e4Z6+1NPhSuugEsvhc8/T3dUZXRlrYhUJqlEb2YjzGylma02s31mXDWzy81sqZktNrOXzSw/2p5nZp9F2xeb2d2pfgF1pU0beOopuPFG+P3vYcgQWJ+Wac4T05W1IlKRKhO9meUAM4DTgXxgXEkijzPL3Xu7e1/gNmBa3L417t43Wi5PVeDp0KAB3HIL/PWvsHx5qLf/97/THZWISOWSOaMfAKx297XuvguYDYyKL+Dun8SttgAyrBY7tUaPDvX2rVvDSSfBjBmZV28vIlIimUTfEXg/br0w2rYXM7vSzNYQzuivitvV1cxeN7OXzGxwoicwswlmVmBmBUVFRdUIP33y8+HVV2HECJg4ES6+OLPq7atLV9aKZK+UNca6+wx3PxL4EXBjtHkj0MXd+wHXALPM7KAEx97r7jF3j+Xm5qYqpFrXujU8+STcdFMY+XLw4FA/Xt/oylqR7JZMol8PdI5b7xRtq8hs4BsA7v6Fu2+J7i8C1gBH71+omalBA/jpT+GJJ2DlynBl7R13pG9s+/2hK2tFslsyiX4h0M3MuppZY2AsMDe+gJl1i1s9E1gVbc+NGnMxsyOAbsDaVASeaUaNgjffDL1xrr46DHu8bFm6o0qOrqwVyW5VJnp3LwYmAvOAFcAcd19mZpPNbGRUbKKZLTOzxYQqmgui7ScCS6LtjwGXu/tHKX8VGaJLlzB0wsyZsHo19OsHN9+c+WPl6MpakexmnmHdRWKxmBcUFKQ7jBorKoLvfz8k/fz80Pd+4MB0R5VYSR19fPVN8+a66EqkPjGzRe4eS7RPV8bWktxcePhhePpp2L4d/uu/4Hvfgx070h3ZvlJxZa167YhkLp3R14Ht2+G66+C3vw3VIffcA6edlu6oUke/CETST2f0adaqFfzmN+Eq2mbNQt/788+HLVvSHVlqqNeOSGZToq9DgwbB66+H8XIeeQR69IBHH63/V9Wq145IZlOir2NNm4bxchYtCnXhY8eGrpmFhemObP+p145IZlOiT5M+feD//g9uvx3+8Y/QM+fuu8Pok/VNKsbDV2OuSO1Rok+jhg3hBz8IM1h99athrPuBA+vfiJg17bWjIRhEapd63WQI99Ad87rrwjj3o0fDrbdCt25VH1vf5eWF5F7e4YfXz7GDRNJBvW7qATP4znfg7bdDHf5zz4XqnKuvho+y9lriQI25IrVLiT7DNG8eeuWsXg0XXQR33QVHHgnTpmX+UAr7KxWNuarjF6mYEn2GOuSQUM/9xhuh3v4HPwhn+H/+c/3vjlleTRtzVccvUjkl+gzXqxc8+yzMmwctWsA554SRMV95Jd2RpU5NG3N1wZZI5dQYW4/s3h0mOPnxj+GDD+Db34Zf/AK6dk13ZOnVoEHiXzlm9bO7qsj+UGNslsjJgUsugVWrQrKfOxe6d4cf/hC2bk13dOmjC7ZEKqdEXw+1bAmTJ4ceOueeGy66Ouqo0HD75Zfpjq7upeKCLZFspkRfj3XqFKpyFi2CY4+Fq66Cnj3hr3/NvgbbymiYZZHKJZXozWyEma00s9VmNinB/svNbKmZLTazl80sP27fddFxK80siwbnzRz9+oVhFJ56Cho1gu5YSA8AABEFSURBVLPPDg22//lPuiOrO+PHh4ur9uwJt9VN8uq1I9msykQfzfk6AzgdyAfGxSfyyCx37+3ufYHbgGnRsfmEOWZ7AiOA35bMISupZQZf/3rojnnfffDOO2G0zLPPDnX6UrFU9NrRLwLJZMmc0Q8AVrv7WnffBcwGRsUXcPdP4lZbACUVB6OA2e7+hbu/A6yOHk9qScOGZQ22kyeXXWE7cSJ8+GG6o8tMNb0yV78IJNMlk+g7Au/HrRdG2/ZiZlea2RrCGf1V1TlWUq9Fi9AzZ/VquPTSMDLmUUeFBsryZ68Hupr22lE/fsl0KWuMdfcZ7n4k8CPgxuoca2YTzKzAzAqKiopSFZIABx8cpjB88004+eQwvEK3bvDAA6FfvtS8147G6pFMl0yiXw90jlvvFG2ryGzgG9U51t3vdfeYu8dyc3OTCEmqq3t3ePzxMARyly5w8cXQt2+46vZA6qGTSE177agfv2S6ZBL9QqCbmXU1s8aExtW58QXMLH4w3TOBkua/ucBYM2tiZl2BbsCrNQ9b9ldJb5w//xk++wzOOANOPRVeey3dkaVXTXrtaOIVyXRVJnp3LwYmAvOAFcAcd19mZpPNbGRUbKKZLTOzxcA1wAXRscuAOcBy4O/Ale6uCoM0M4MxY2D5crjzTli8GI47LgyTvGCBqnSqSxOvSKbTWDfCtm1hkpNf/xo+/xzatYNTToHhw+G008KFWVJ7NPGKpILGupFKtW4NP/95mKD8kUdg5MhQl3/JJdC5c7ja9vvfh7//XT12akMqGnNV9SOV0Rm9JOQeeuo891wYInn+/DDxSZMmcOKJZWf7vXqF6grZfzU9oy+p+on/Em7evPrDQEj9VtkZvRK9JGXnznCWP29eWJYvD9sPPbQs6Z96KnTokN4466OaJmpV/Qio6kZSoHnzkMynTYNly0K1wu9/D4MHh+GSzz0XvvIVGDoU/vAH2L493RHXHzVtzFXVj1RFZ/RSY7t3Q0FB6JM/c2a4Grd58zDOzgUXwLBhIYFI7VDVj4DO6KWW5eTA8cfDzTeHMfJffjkkiCefDL138vLCcABvv53uSLNTTfvxawiH7KdELyllFkbNvPfeMN3hI4+EXjtTp8Ixx8DXvgb33HNgz4iVaplQ9SOZTYleak2zZjB2bKjSef99uO22UHd/+eVwyCFhzttnnoHi4nRHWv/V5MreTBjCQW0EtUuJXurEYYfB//wPLF0KCxeGETX/+U8488zQV//aa8M+qXvpHsJBVwbXPjXGStrs2gVPPw0PPRRui4vhiCNCP/0hQ8Jt167qp18XZs4MdfLvvRfO5KdMqf4QDuoeml7qRy8Zr6gI5swJZ/nz58OWLWF7p057J/5jjlHizzQ1TdQNGiQeQdUsVEXVhZp80WUK9bqRjJebC1deGSY2//DDcFXujBmhYfeFF+Cyy6BHj1C3/61vwV13wZIldZcIpGI1bcxNRRuBqo4qpzN6yXjuoW/+Sy+Fs/2XXipLIm3ahIu2Ss74+/YNE6RL3Ul3P35VHQWqupGs8+67ZUl//vyyCdCbNAnJPhYLQy/HYuGXQMOG6Y03m6XigquaVJ1kQ9VRKijRS9bbsCGMxbNwYbhKd9Ei2LEj7GvWDPr1K0v8sVio68/JSW/M2SSdddw1TdSpOKPPhDp+JXo54OzZE67EXbQoJP6CgjCLVslZZ4sWIfmXJP5YLMylq6Ea6p/6XnWUKkr0IoQxeVauLEv8BQVhdq3PPgv7W7UK1TxHHrnvcuih6u2Tqep71VGq1DjRm9kI4A4gB7jf3aeW238NcAlQDBQB33X3d6N9u4GSS2Hec/eRVEKJXupScTGsWFGW+FeuhDVrwj98/M/+Zs0SfwEceWT4h1YDcHrV56ojSE38NUr0ZpYDvA2cChQSJgsf5+7L48oMAxa4+04zuwIY6u7fjvbtcPeWyQarRC+ZYNeucJa2Zs2+y9q1Zb8CINT1d+kSkn5+Phx7bGgQzs+Hpk3T9xqkbqS76qhETRP914Cb3f20aP06AHf/RQXl+wG/cfdB0boSvWQVd9i4cd8vgNWrw4Qsn34ayuXkQPfuIekfe2zZF8BXvpLe+CW1MqV7Z2WJPplOZx2B9+PWC4HjKyl/MfBs3HpTMysgVOtMdfcnEgQ4AZgA0KUuR1IS2Q9mYeyeww4Lffjj7dkTkv7ixfDGG2F56aW9L7455JCypF9y262buoDWVyXJfH+rXupi9NBkzujHACPc/ZJo/TvA8e4+MUHZ84CJwBB3/yLa1tHd15vZEcALwMnuvqai59MZvWSjLVvKEv8bb4QvguXL4csvw/6mTcP8u7EYDBwYxvc/+mj1AjoQZMoZ/Xqgc9x6p2hb+Sc5BbiBuCQP4O7ro9u1ZvYvoB9QYaIXyUbt28NJJ4WlxK5doSG4JPm//jrMmgV33x32t2kTEn5J4j/+eGjXLj3xS+2ZMiVx1U91Rg+tSjKJfiHQzcy6EhL8WODc+AJRvfw9hDP/D+O2twV2uvsXZtYBGATclqrgReqzxo3L6u5L7NkDb70Fr7xStkyeXNar4+ijQ+IvWXr3VpVPfVfTqp9kJNu98gxgOqF75QPuPsXMJgMF7j7XzP4B9AY2Roe85+4jzey/CF8AewgDqE13999X9lyquhHZ2/btoetnfPL/MDqdatasrLpn4MBQ33/44brq90CkC6ZEsoh7qLtdsKAs8b/+eqgKglDff/TR4eKv7t3Lbo8+OnwxSHZSohfJcl98ERp4ly4NVT8rVoRl3bqyah+zMJFLfPIvuW3fPq3hSwrUtDFWRDJckyZlDbbxPvssjPlTkvxLbv/5z/DlUCI3NyT8nj1DvX+fPqEXUJs2dfs6pHbojF7kALR7d+jSV/4L4M03Ydu2snKdO5cl/t69w3LMMaEhWTKLzuhFZC85OWF+3iOOgDPOKNvuDoWFoQpoyZJwu3QpPPdcGBcIwrg+3buXJf6SL4JOnTTwW6ZSoheRUmbhLL5z572/AHbtCgO+lST+JUvC+P+zZpWVad06jPdz8MGVL+3b60KwuqZELyJVaty47Ow93tatobqn5Oz/vfdg06Zwf9Omsit/4+XkhDaBgw8Ow0HEfwl07BgajLt2DWX0CyE1lOhFZL+1aQMnnBCW8tzh449Dwq9sWbEi3MY3DkO4OjQvLyT9ktv4+23b1sELzBJK9CJSK8zCkA3t2oVunJVxD43AhYXwzjthWbeu7Pbll/duJIZQVVT+C6Bjx/BLoGRp21YXj4ESvYhkALPw66BNm9CtM5GPPy5L/vFfBG+/HRqL48eKKdGgQfiiyc2FDh32/hKIXy+5f/DB2TmJjBK9iNQLbduGpV+/ffe5Q1FRmCegqKhs2bx57/W33gqNyFu2JJ79KScn/DLo1m3v5aijwvb6Oq5QPQ1bRKSMWZjQJdlJXXbvDg3J8V8CRUWhMXn1ali1KlQX7dhRdkzDhqGKqPwXQLdumT++kBK9iBxwcnJCN8/27cM1AYm4h0biVavKkn/J8tJLZTOJQajuKbku4fDDy5YuXcLtoYem94tAiV5EJAGz0P3zkEP2nUnMHT74YO/kv2pVaDNYsAA++mjv8g0bhmsT4pN//NK5c+3OL6xELyJSTWbhLP3QQ+HEE/fdv2NHqAZ6992ypWT9hRdgw4Z92wgOPhiGDoXZs1MfrxK9iEiKtWwJ+flhSeTLL2H9+n2/CHJzayceJXoRkTrWqFHoxZOXVzfPpxEnRESyXFKJ3sxGmNlKM1ttZpMS7L/GzJab2RIz+6eZHR637wIzWxUtF6QyeBERqVqVid7McoAZwOlAPjDOzMrXPL0OxNy9D/AY0QTgZtYO+AlwPDAA+Ek0YbiIiNSRZM7oBwCr3X2tu+8CZgOj4gu4+4vuXnIB8itAp+j+acDz7v6Ru38MPA+MSE3oIiKSjGQSfUfg/bj1wmhbRS4Gnq3OsWY2wcwKzKygqKgoiZBERCRZKW2MNbPzgBjwy+oc5+73unvM3WO5tdW/SETkAJVMol8PdI5b7xRt24uZnQLcAIx09y+qc6yIiNSeZBL9QqCbmXU1s8bAWGBufAEz6wfcQ0jyH8btmgcMN7O2USPs8GibiIjUEXP3qguZnQFMB3KAB9x9iplNBgrcfa6Z/QPoDWyMDnnP3UdGx34XuD7aPsXdH6ziuYqAd/fr1QQdgM01OL62Kb6aUXw1o/hqJpPjO9zdE9Z9J5Xo6xMzK3D3WLrjqIjiqxnFVzOKr2YyPb6K6MpYEZEsp0QvIpLlsjHR35vuAKqg+GpG8dWM4quZTI8voayroxcRkb1l4xm9iIjEUaIXEcly9TLRJzFschMzezTav8DM8uowts5m9mI0bPMyM/tegjJDzWybmS2OlpvqKr64GNaZ2dLo+QsS7DczuzN6D5eYWf86jO2YuPdmsZl9YmZXlytTp++hmT1gZh+a2Ztx29qZ2fPRENzPVzQya10M1V1BfL80s7eiv9/jZtamgmMr/SzUYnw3m9n6uL/hGRUcW+n/ey3G92hcbOvMbHEFx9b6+1dj7l6vFsJFW2uAI4DGwBtAfrky/w+4O7o/Fni0DuM7FOgf3W8FvJ0gvqHA39L8Pq4DOlSy/wzC4HQGDAQWpPHv/QHhYpC0vYfAiUB/4M24bbcBk6L7k4BbExzXDlgb3baN7reto/iGAw2j+7cmii+Zz0ItxnczcG0Sf/9K/99rK75y+38F3JSu96+mS308o69y2ORo/aHo/mPAyWZmdRGcu29099ei+9uBFVQ+2memGgX80YNXgDZmdmga4jgZWOPuNblausbcfT7wUbnN8Z+zh4BvJDi0TobqThSfuz/n7sXRavzw4XWugvcvGcn8v9dYZfFFueMc4JFUP29dqY+JPpmhj0vLRB/0bUD7OokuTlRl1A9YkGD318zsDTN71sx61mlggQPPmdkiM5uQYH91h6euLWOp+B8s3e/hwe5eMuzHB8DBCcpkyvv4XcqGDy+vqs9CbZoYVS09UEHVVya8f4OBTe6+qoL96Xz/klIfE329YGYtgb8AV7v7J+V2v0aoijgWuAt4oq7jA05w9/6EmcOuNLMT0xBDpSwMojcS+HOC3ZnwHpby8Bs+I/sqm9kNQDEws4Ii6fos/A44EuhLGCfrV3X0vNU1jsrP5jP+f6k+Jvpkhj4uLWNmDYHWwJY6iS48ZyNCkp/p7n8tv9/dP3H3HdH9Z4BGZtahruKLnnd9dPsh8DjhJ3K8TBhi+nTgNXffVH5HJryHwKaS6qzo9sMEZdL6PprZhcDXgfHRl9E+kvgs1Ap33+Tuu919D3BfBc+b7vevIfBN4NGKyqTr/auO+pjoqxw2OVov6d0wBnihog95qkX1eb8HVrj7tArKHFLSZmBmAwh/h7r8ImphZq1K7hMa7d4sV2wucH7U+2YgsC2umqKuVHgmle73MBL/ObsAeDJBmbQN1W1mI4AfEoYP31lBmWQ+C7UVX3ybz+gKnjeZ//fadArwlrsXJtqZzvevWtLdGrw/C6FHyNuE1vgbom2TCR9ogKaEn/urgVeBI+owthMIP+GXAIuj5QzgcuDyqMxEYBmhB8ErwH/V8ft3RPTcb0RxlLyH8TEaYVL4NcBSwuTvdRljC0Libh23LW3vIeELZyPwJaGe+GJCu88/gVXAP4B2UdkYcH/csd+NPourgYvqML7VhPrtks9hSU+0w4BnKvss1FF8f4o+W0sIyfvQ8vFF6/v8v9dFfNH2P5R85uLK1vn7V9NFQyCIiGS5+lh1IyIi1aBELyKS5ZToRUSynBK9iEiWU6IXEclySvQiIllOiV5EJMv9f51VuX4GuUT0AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "accuracy = sport_train.history['accuracy']\n",
    "val_accuracy = sport_train.history['val_accuracy']\n",
    "loss = sport_train.history['loss']\n",
    "val_loss = sport_train.history['val_loss']\n",
    "epochs = range(len(accuracy))\n",
    "plt.plot(epochs, accuracy, 'bo', label='Training accuracy')\n",
    "plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')\n",
    "plt.title('Training and validation accuracy')\n",
    "plt.legend()\n",
    "plt.figure()\n",
    "plt.plot(epochs, loss, 'bo', label='Training loss')\n",
    "plt.plot(epochs, val_loss, 'b', label='Validation loss')\n",
    "plt.title('Training and validation loss')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:20:58.050602Z",
     "start_time": "2018-11-08T00:20:55.021862Z"
    }
   },
   "outputs": [],
   "source": [
    "predicted_classes2 = sport_model.predict(test_X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:20:58.262575Z",
     "start_time": "2018-11-08T00:20:58.052878Z"
    }
   },
   "outputs": [],
   "source": [
    "predicted_classes=[]\n",
    "for predicted_sport in predicted_classes2:\n",
    "    predicted_classes.append(predicted_sport.tolist().index(max(predicted_sport)))\n",
    "predicted_classes=np.array(predicted_classes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:20:58.272559Z",
     "start_time": "2018-11-08T00:20:58.264703Z"
    }
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((5158,), (5158,))"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "predicted_classes.shape, test_Y.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Aprendamos de los errores: Qué mejorar"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:20:59.822110Z",
     "start_time": "2018-11-08T00:20:58.275464Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 4805 correct labels\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 9 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "correct = np.where(predicted_classes==test_Y)[0]\n",
    "print(\"Found %d correct labels\" % len(correct))\n",
    "for i, correct in enumerate(correct[0:9]):\n",
    "    plt.subplot(3,3,i+1)\n",
    "    plt.imshow(test_X[correct].reshape(21,28,3), cmap='gray', interpolation='none')\n",
    "    plt.title(\"{}, {}\".format(deportes[predicted_classes[correct]],\n",
    "                                                    deportes[test_Y[correct]]))\n",
    "\n",
    "    plt.tight_layout()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:21:00.942267Z",
     "start_time": "2018-11-08T00:20:59.829572Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 353 incorrect labels\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 8 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "incorrect = np.where(predicted_classes!=test_Y)[0]\n",
    "print(\"Found %d incorrect labels\" % len(incorrect))\n",
    "for i, incorrect in enumerate(incorrect[0:9]):\n",
    "    plt.subplot(3,3,i+1)\n",
    "    plt.imshow(test_X[incorrect].reshape(21,28,3), cmap='gray', interpolation='none')\n",
    "    plt.title(\"{}, {}\".format(deportes[predicted_classes[incorrect]],\n",
    "                                                    deportes[test_Y[incorrect]]))\n",
    "    plt.tight_layout()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2018-11-08T00:21:00.968727Z",
     "start_time": "2018-11-08T00:21:00.947262Z"
    },
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "     Class 0       0.98      0.89      0.93      1479\n",
      "     Class 1       0.87      0.91      0.89      1865\n",
      "     Class 2       0.91      0.94      0.93      1814\n",
      "\n",
      "    accuracy                           0.92      5158\n",
      "   macro avg       0.92      0.91      0.92      5158\n",
      "weighted avg       0.92      0.92      0.92      5158\n",
      "\n"
     ]
    }
   ],
   "source": [
    "target_names = [\"Class {}\".format(i) for i in range(nClasses)]\n",
    "print(classification_report(test_Y, predicted_classes, target_names=target_names))"
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
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
