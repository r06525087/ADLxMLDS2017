
# coding: utf-8

# In[38]:


import numpy as np
import json
import sys

path_1 = sys.argv[1]
path_2 = sys.argv[2]
path_3 = sys.argv[3]



with open(path_1+'testing_label.json') as data_file:    
    test_label = json.load(data_file)

    
test_data = []

test_file_names = ['klteYv1Uv9A_27_33.avi','5YJaS2Eswg0_22_26.avi',
                   'UbmZAe5u5FI_132_141.avi','JntMAcTlOF0_50_70.avi','tJHUH9tpqPg_113_118.avi']

for i in range(len(test_file_names)):

    test_data.append(np.load(path_2+'testing_data/feat/'+test_file_names[i]+'.npy'))


# In[39]:


def result_to_result(input_list):  #丟入一個句子轉為真的句子
    sentence=input_list[0]+' '
    
    for i in range(1,len(input_list)):
        if input_list[i] != '.':
            sentence += input_list[i]+' '
    sentence=sentence[:-1]+'.'
    
    return sentence
    


# In[40]:


num_to_words =  np.load('test_dictionary.npy').item()

def result_to_words(result):
    idxs=[]
    for i in range(len(result)):
        idx = list(result[i]).index(max(list(result[i])))
        idxs.append(idx)
    
    sentences=[]
    for k in range(len(idxs)):
        sentences.append(num_to_words[str(idxs[k])])
    
    return sentences


# In[41]:


test_data[0].shape


# In[42]:


from keras.models import load_model
model = load_model('my_model.h5')


# In[43]:


import tensorflow as tf
import numpy as np
import keras
from keras.layers import GRU,Reshape,Multiply,MaxPooling2D,Flatten,Conv2D,Concatenate,Masking,Add,concatenate, UpSampling1D,MaxPooling1D,Input,Dense,LSTM,TimeDistributed,Activation,Dropout,BatchNormalization,Conv1D,Bidirectional
from keras.models import Model, Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.core import RepeatVector
from keras.layers import convolutional
test_predict = model.predict(x=np.array(test_data))


# In[44]:


print(result_to_words(test_predict[0]))


# In[45]:


really_result=[]
for i in range(0,5):
    the_sentence = result_to_result(result_to_words(test_predict[i]))
    really_result.append(the_sentence)


# In[46]:


print(really_result)


# In[47]:


the_result_list=[]

for i in range(0,5):
    item = test_file_names[i]+','+really_result[i]+'\n'

    the_result_list.append(item)
    


# In[48]:


print(the_result_list)


# In[49]:


with open(path_3, mode='wt', encoding='utf-8') as myfile:
    for i in range(0,5):
        myfile.write(the_result_list[i])


# In[50]:


watch_my_example=[]
file = open(path_3)

for line in file:
    watch_my_example.append(line)
print(watch_my_example)

