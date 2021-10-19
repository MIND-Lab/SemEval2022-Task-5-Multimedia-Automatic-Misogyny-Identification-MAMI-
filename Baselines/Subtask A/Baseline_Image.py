#pip install -r requirements.txt
'''
#requires the same folder:
- script evaluation
-folder 'ref' with truth.txt
-folder 'TRAINING' with images
'''

#path
csv_path_test = './test.csv'
csv_path_train = './train.csv'
image_path = './TRAINING'

import evaluation
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import regularizers
import os
import gc
import shutil 

def loadImage(image_path):
    try:
        return load_img(image_path, target_size=(image_size, image_size))
    except:
        image_path = image_path.replace('png', 'jpg')
        return load_img(image_path, target_size=(image_size, image_size))

if not os.path.exists('./ImageModel'):
    os.makedirs('./ImageModel')
    
batch_size = 32
epochs = 50
image_size = 224
threshold = 0.5

tf.compat.v1.disable_eager_execution()

#_______________________________Load Train Data_______________________________
train_df = pd.read_csv(csv_path_train, usecols=['file_name', 'misogynous', 'Text Transcription'], sep='\t')
path = image_path+'/'
train_df['image_path'] = path + train_df['file_name']

#load images
train_df['image'] = None
train_df['image'] = train_df['image_path'].apply(lambda x: img_to_array(loadImage(x)))        

#division and processing of data as input to the model
X_train = train_df[['file_name', 'image']]
y_train = train_df['misogynous']

#images
tmp = []
for value in X_train['image']:
  tmp.append(value) 
iX_train = np.array(tmp)

#misogynous label
tmp = []
for value in y_train:
    tmp.append([value])  
y_train = np.array(tmp)

#clear memory
del train_df

gc.collect()

#_______________________________IMAGE MODEL_______________________________
l2_strength = 1e-5

input_image = layers.Input(shape=(image_size,image_size,3))
vgg_model = VGG16(input_tensor = input_image, weights = 'imagenet', include_top=False)

for layer in vgg_model.layers:
    layer.trainable = False

x = vgg_model.output
x = layers.Flatten(input_shape=vgg_model.output_shape[1:])(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength))(x)
x = layers.Dense(1, activation='sigmoid')(x)
image_model = Model(vgg_model.input, x)

image_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
image_model.summary()

history = image_model.fit(iX_train, 
                    y_train,
                    validation_split=0.1,
                    epochs= epochs,
                    batch_size=batch_size,
                    #verbose=0,
                    )
                    
image_model.save('ImageModel/model_image.h5')
tf.keras.utils.plot_model(image_model, "./ImageModel/model_image.png", show_shapes=True)

#_______________________________Load Test Data_______________________________
#clear memory
del X_train
del y_train
del iX_train

gc.collect()

#load data
test_df = pd.read_csv(csv_path_test, sep='\t')
path = image_path+'/'
test_df['image_path'] = path + test_df['file_name']

#Load images
test_df['image'] = None
test_df['image'] = test_df['image_path'].apply(lambda x: img_to_array(loadImage(x)))        

#division and processing of data as input to the model
#images
tmp = []
for value in test_df['image']:
    tmp.append(value)  
iX_test = np.array(tmp)

#_______________________________PREDICTION_______________________________
predictions = image_model.predict(iX_test, batch_size=batch_size)
predictions = predictions.reshape(predictions.shape[0])
pred = predictions > threshold
pred = list(map(int, pred)) #true/false to 1/0

predictions_db = pd.DataFrame(data=test_df['file_name'])
predictions_db['misogynist'] = pred

#_______________________________EVALUATION_______________________________
if not os.path.exists('./res'):
    os.makedirs('./res')

predictions_db.to_csv('./res/answer.txt', index=False, sep='\t', header=False)  
evaluation.main(['','./', './ImageModel'])
#move res folder to ImageModel folder
shutil.move('./res/', './ImageModel')
