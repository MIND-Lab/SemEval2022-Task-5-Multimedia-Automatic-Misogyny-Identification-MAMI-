#pip install -r requirements.txt
'''
requires the same folder:
- script evaluation
-folder 'ref' with truth.txt
'''

#path
csv_path_test = './test.csv'
csv_path_train = './train.csv'

import evaluation
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import keras
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import os
import gc
import shutil 

if not os.path.exists('./TextModel'):
    os.makedirs('./TextModel')
    
batch_size = 32
epochs = 50
embed_size = 512 #according to USE
threshold = 0.5

#Universal Sentence Encoder
tf.compat.v1.disable_eager_execution()

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed"
embed = hub.Module(module_url)

#_______________________________Load Training Data_______________________________
train_df = pd.read_csv(csv_path_train, usecols=['file_name', 'misogynous', 'Text Transcription'], sep='\t')

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

#division and processing of data as input to the model
X_train = train_df[['file_name', 'USE']]
y_train = train_df['misogynous']

#text
tmp = []
for value in X_train['USE']:
    tmp.append([value])  
tX_train = np.array(tmp)

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

#_______________________________TEXT MODEL_______________________________
input_text = layers.Input(shape=(1, embed_size))
l = layers.Dense(1, activation='sigmoid')(input_text)
model = Model(inputs=[input_text], outputs=l)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(tX_train, 
                    y_train,
                    validation_split=0.1,
                    epochs= epochs,
                    batch_size=batch_size,
                    verbose=0)
                    
model.save('./TextModel/model_text.h5')
tf.keras.utils.plot_model(model, "./TextModel/model_text.png", show_shapes=True)

#_______________________________Load Test Data_______________________________
#clear memory
del X_train
del y_train
del tX_train

gc.collect()

test_df = pd.read_csv(csv_path_test, sep='\t')

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

#division and processing of data as input to the model
#Text
tmp = []
for value in test_df['USE']:
    tmp.append([value])  
tX_test = np.array(tmp)

#clear memory
del dfs
del text_embedding
del text_embeddings

gc.collect()

#_______________________________PREDICTION_______________________________
predictions = model.predict(tX_test, batch_size=batch_size)
predictions = predictions.reshape(predictions.shape[0])
pred = predictions > threshold
pred = list(map(int, pred)) #true/false to 1/0

predictions_db = pd.DataFrame(data=test_df['file_name'])
predictions_db['misogynist'] = pred

#_______________________________EVALUATION_______________________________
if not os.path.exists('./res'):
    os.makedirs('./res')
    
predictions_db.to_csv('./res/answer.txt', index=False, sep='\t', header=False)  
evaluation.main(['','./', './TextModel/'])
#move res folder to TextModel folder
shutil.move('./res/', './TextModel/res/') 
