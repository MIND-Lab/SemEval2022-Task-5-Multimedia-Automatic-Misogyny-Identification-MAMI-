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

if not os.path.exists('./HierarchicalMultilabelModel'):
    os.makedirs('./HierarchicalMultilabelModel')
    
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
train_df = pd.read_csv(csv_path_train, sep='\t')
 
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
   
#clear memory
del dfs
del text_embedding
del text_embeddings

gc.collect()

#division and processing of data as input to the model
X_train = train_df[['file_name', 'USE']]
y_train = train_df[['misogynous','shaming','stereotype','objectification','violence']]

#text
tmp = []
for value in X_train['USE']:
    tmp.append([value])  
tX_train = np.array(tmp)

#misogynous label
tmp = []
for value in y_train['misogynous']:
    tmp.append([value])  
y_train = np.array(tmp)

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

text_model.save('./HierarchicalMultilabelModel/text_model.h5')
tf.keras.utils.plot_model(text_model, "./HierarchicalMultilabelModel/text_model.png", show_shapes=True)

#clear memory
del tmp
del X_train
del y_train
del tX_train

gc.collect()

#division and processing of data as input to the model
train_df_Mis = train_df[train_df['misogynous']==1]
X_train_Mis = train_df[['file_name', 'USE']]
y_train_Mis = train_df[['shaming','stereotype','objectification','violence']]

#text
tmp = []
for value in X_train_Mis['USE']:
    tmp.append([value])  
tX_train_Mis = np.array(tmp)

#misogynous label
tmp = []
for index, row in y_train_Mis.iterrows():
  tmp.append(np.array(row.values))
y_train_Mis = np.array(tmp)

#_______________________________MISOGYNY TYPE TEXT MODEL_______________________________
input_text = layers.Input(shape=(1, embed_size))
l = layers.Flatten(input_shape=(1, embed_size)[1:])(input_text)
l = layers.Dense(4, activation='sigmoid')(l)
mis_text_model = Model(inputs=[input_text], outputs=[l])
mis_text_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
mis_text_model.summary()

history = mis_text_model.fit(tX_train_Mis, 
                    y_train_Mis,
                    validation_split=0.1,
                    epochs= epochs,
                    batch_size=batch_size,
                    #verbose=0
                    )

mis_text_model.save('./HierarchicalMultilabelModel/text_model_mis.h5')
tf.keras.utils.plot_model(mis_text_model, "./HierarchicalMultilabelModel/mis_text_model.png", show_shapes=True)

#clear memory
del tmp
del train_df_Mis
del X_train_Mis
del y_train_Mis
del tX_train_Mis

gc.collect()

#_______________________________Load Test Data_______________________________
#load data
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

#clear memory
del dfs
del text_embedding
del text_embeddings

gc.collect()

#division and processing of data as input to the model
#Text
tmp = []
for value in test_df['USE']:
    tmp.append([value])  
tX_test = np.array(tmp)

#_______________________________PREDICTION_______________________________
predictions = text_model.predict(tX_test, batch_size=batch_size)
predictions = predictions.reshape(predictions.shape[0])
pred = predictions > threshold
pred = list(map(int, pred)) #true/false to 1/0

predictions_db = pd.DataFrame(data=test_df['file_name'])
predictions_db['misogynous'] = pred

list_mis = predictions_db.loc[predictions_db['misogynous']==1,'file_name'].tolist()
test_df_Mis = test_df.loc[test_df['file_name'].isin(list_mis), :]

mis_predictions = mis_text_model.predict(tX_test, batch_size=batch_size)
mis_predictions_db = pd.DataFrame(mis_predictions,  columns=['shaming','stereotype','objectification','violence'])
mis_predictions_db = mis_predictions_db.apply(lambda x:  list(map(int, (x > threshold))))
mis_predictions_db['file_name']=test_df_Mis['file_name']

mis_predictions_db = mis_predictions_db[['file_name', 'shaming','stereotype','objectification','violence']]

for x in ['shaming','stereotype','objectification','violence']:
  predictions_db[x]=0
  
for mis_file in list_mis:
  for mis_type in ['shaming','stereotype','objectification','violence']:
    predictions_db.loc[predictions_db['file_name']== mis_file, mis_type] = mis_predictions_db.loc[mis_predictions_db['file_name']== mis_file, mis_type]
    
#_______________________________EVALUATION_______________________________
if not os.path.exists('./res'):
    os.makedirs('./res')

predictions_db.to_csv('./res/answer.txt', index=False, sep='\t', header=False)
evaluation.main(['','./', './HierarchicalMultilabelModel/'])
#move res folder to ConcatenatedModel folder
shutil.move('./res/', './HierarchicalMultilabelModel/res/')
