#pip install -r requirements.txt
'''
requires the same folder:
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

if not os.path.exists('./ImageTextModel'):
    os.makedirs('./ImageTextModel')
    
batch_size = 32
epochs = 50
image_size = 224
embed_size = 512 #according to USE
threshold = 0.5

#Universal Sentence Encoder
tf.compat.v1.disable_eager_execution()

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed"
embed = hub.Module(module_url)

#_______________________________Load Train Data_______________________________
train_df = pd.read_csv(csv_path_train, usecols=['file_name', 'misogynous', 'Text Transcription'], sep='\t')
path = image_path+'/'

train_df['image_path'] = path + train_df['file_name']

# Universal Sentence Encoder (USE)
'''
Split the dataset to avoid hitting the USE call limit
an error occurs if the 47900 steps are reached
'''
dfs = np.array_split(train_df, 10)
train_df['USE'] = None
text_embeddings=[]
with tf.compat.v1.Session() as session:
    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
   
    for x in dfs:
      x_data_matches = pd.Series([])
      text_embedding = session.run(embed(list(x['Text Transcription'])))
      text_embeddings = text_embeddings + np.array(text_embedding).tolist()
      
train_df['USE'] = text_embeddings

#load images
train_df['image'] = None
train_df['image'] = train_df['image_path'].apply(lambda x: img_to_array(loadImage(x)))        

#division and processing of data as input to the model
X_train = train_df[['file_name', 'USE', 'image']]
y_train = train_df['misogynous']

#text
tmp = []
for value in X_train['USE']:
    tmp.append([value])  
tX_train = np.array(tmp)

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
del dfs
del text_embedding
del text_embeddings

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
                    
image_model.save('./ImageTextModel/image_model.h5')

#_______________________________TEXT MODEL_______________________________
input_text = layers.Input(shape=(1, embed_size))
l = layers.Dense(1, activation='sigmoid')(input_text)
text_model = Model(inputs=[input_text], outputs=l)
text_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
text_model.summary()

history = text_model.fit(tX_train, 
                    y_train,
                    validation_split=0.1,
                    epochs= epochs,
                    batch_size=batch_size,
                    #verbose=0
                    )

text_model.save('./ImageTextModel/text_model.h5')

#_______________________________IMAGE-TEXT MODEL_______________________________
image = image_model.layers[len(image_model.layers)-2].output
reshape = layers.Reshape((1, image_model.layers[len(image_model.layers)-2].output_shape[1]), name='predictions')(image)
text = text_model.layers[0].output

input = tf.keras.layers.Concatenate(axis=-1)([text, reshape])

l = layers.Dense(1, activation='sigmoid')(input)
model = Model(inputs=[input_text, input_image], outputs=[l])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.save('./ImageTextModel/image_text_model.h5')
tf.keras.utils.plot_model(model, "./ImageTextModel/model.png", show_shapes=True)

#_______________________________Load Test Data_______________________________
#clear memory
del X_train
del y_train
del iX_train
del tX_train

gc.collect()

#load data
test_df = pd.read_csv(csv_path_test, usecols=['file_name', 'misogynous', 'Text Transcription'], sep='\t')
path = image_path+'/'
test_df['image_path'] = path + test_df['file_name']

# Universal Sentence Encoder (USE)
'''
Split the dataset to avoid hitting the USE call limit
an error occurs if the 47900 steps are reached
'''
dfs = np.array_split(test_df, 10)
test_df['USE'] = None
text_embeddings=[]
with tf.compat.v1.Session() as session:
    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
   
    for x in dfs:
      x_data_matches = pd.Series([])
      text_embedding = session.run(embed(list(x['Text Transcription'])))
      text_embeddings = text_embeddings + np.array(text_embedding).tolist()
test_df['USE'] = text_embeddings

#Load images
test_df['image'] = None
test_df['image'] = test_df['image_path'].apply(lambda x: img_to_array(loadImage(x)))        

#division and processing of data as input to the model
X_test = test_df[['file_name', 'USE', 'image']]
y_test = test_df['misogynous']

#Text
tmp = []
for value in X_test['USE']:
    tmp.append([value])  
tX_test = np.array(tmp)


#images
tmp = []
for value in X_test['image']:
    tmp.append(value)  
iX_test = np.array(tmp)

#misogynous label
tmp = []
for value in y_test:
    tmp.append([value])  
y_test = np.array(tmp)

#clear memory
del tmp
del dfs
del text_embedding
del text_embeddings

gc.collect()

#_______________________________PREDICTION_______________________________
predictions = model.predict([tX_test, iX_test], batch_size=batch_size)
predictions = predictions.reshape(predictions.shape[0])
pred = predictions > threshold
pred = list(map(int, pred)) #true/false to 1/0

predictions_db = pd.DataFrame(data=test_df['file_name'])
predictions_db['misogynist'] = pred

#_______________________________EVALUATION_______________________________
if not os.path.exists('./res'):
    os.makedirs('./res')

predictions_db.to_csv('./res/answer.txt', index=False, sep='\t', header=False)
evaluation.main(['','./', './ImageTextModel/'])
#move res folder to ImageTextModel folder
shutil.move('./res/', './ImageTextModel/res/') 
