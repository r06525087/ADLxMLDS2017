
# coding: utf-8

# In[1]:

import numpy as np
import keras
from keras.layers import Concatenate,Add,Input,Dense,Activation,Conv2DTranspose,Reshape,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential, load_model
from keras import backend as K
import pickle
import numpy as np
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import numpy as np
import matplotlib.image as mtimg
import matplotlib.pyplot as plt
import random as rnd
from PIL import Image
import cv2 as cv
import math
import random
from random import shuffle
def get_the_list(path):
    img_name_list = []
    img_list = []
    for filename in os.listdir(path):
        img_name_list.append(filename)
    for i in range (0,len(img_name_list)):
        if img_name_list[i][-4:]=='.jpg':
            img = mtimg.imread(path+'/'+img_name_list[i])
            img_list.append(img)
    
    return img_list

def save_(fake_imgs,num,file_name):
    
    sampled_fake = random.sample(fake_imgs,num)
    for n in range(num):
        img = sampled_fake[n][0]
        img = img*255
        img = Image.fromarray(img.astype('uint8'))
        img_name = file_name + '__' + str(n) + '.jpg'
        img.save(img_name)
def shuffle_two(list_1, list_2):
    
    c = list(zip(list_1, list_2))
    random.shuffle(c)
    r_1,r_2 = zip(*c)  
    return r_1,r_2
def plot_img(image):
    plt.imshow(image)
    plt.show()
    

def label_to_array(input_tag):
    

    hair_array = hair_condition[input_tag[1]]
    eyes_array = eyes_condition[input_tag[2]]
    array = np.concatenate([hair_array,eyes_array])
    
    return array


# In[2]:

real_imgs = np.load('imgs.npy')


# In[3]:

with open('parrot.pkl', 'rb') as f:
    tags = pickle.load(f)


# In[ ]:

k = 7548
plot_img(real_imgs[k])
print(tags[k])


# In[5]:

hw_hair = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
           'green hair', 'red hair', 'purple hair', 'pink hair',
           'blue hair', 'black hair', 'brown hair', 'blonde hair']
hw_eyes = ['gray eyes', 'black eyes', 'orange eyes','pink eyes', 
           'yellow eyes', 'aqua eyes', 'purple eyes',
           'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

hair_condition = dict()
for i in range(len(hw_hair)):
    key = hw_hair[i]
    value = np.zeros(12)
    value[i] = 1
    hair_condition[key] = value

eyes_condition = dict()
for i in range(len(hw_eyes)):
    key = hw_eyes[i]
    value = np.zeros(11)
    value[i] = 1
    eyes_condition[key] = value


# ## MAKE INPUT NOISE WITH LABEL

# In[6]:

def noise_maker(m, n):
    noise = np.random.uniform(-1,1.,size=[m,n])
    return noise


# In[7]:

fake_input = []

for i in range(len(tags)):
    noise = np.random.uniform(-1,1.,size=[100,])
    
    hair_color = tags[i][1]
    eyes_color = tags[i][2]
    
    hair_color = hair_condition[hair_color]
    eyes_color = eyes_condition[eyes_color]
    
    fake_input.append((noise,hair_color,eyes_color,0))


# In[8]:

fake_input[7548]


# ## MAKE REAL IMG WITH LABEL

# In[9]:

real_imgs = np.array(real_imgs)
real_imgs = real_imgs.astype('float')/255
real_imgs = list(real_imgs)


# In[10]:

real_input = []

for i in range(len(tags)):
    real_img = real_imgs[i]
    
    hair_color = tags[i][1]
    eyes_color = tags[i][2]
    
    hair_color = hair_condition[hair_color]
    eyes_color = eyes_condition[eyes_color]
    
    real_input.append((real_img,hair_color,eyes_color,1))



# In[11]:

plot_img(real_input[7548][0])
print(real_input[7548][1])
real_input[7548][2]


# ## combile(all_model)

# In[12]:

d_model = load_model('d_model.h5')


# In[13]:

g_model = load_model('g_model.h5')


# In[14]:

d_model.summary()
g_model.summary()


# In[15]:

# all_model = Model(inputs = g_model.input,
#                   outputs = ([d_model([g_model.output,g_model.input[1],g_model.input[2]])]))


# In[16]:

# all_model.summary()
# all_model.save('all_model.h5')


# ## TRANING

# In[17]:

def data_pakage(input_datas):
    
    input_1 = []
    input_2 = []
    input_3 = []
    label = []
    
    for i in range(len(input_datas)):
        input_1.append(input_datas[i][0])
        input_2.append(input_datas[i][1])
        input_3.append(input_datas[i][2])
        label.append(input_datas[i][3])
    
    input_1 = np.array(input_1)
    input_2 = np.array(input_2)
    input_3 = np.array(input_3)
    label = np.array(label)
    
    return input_1,input_2,input_3,label


# In[18]:

def fake_data_split(fake_imgs,label_1,label_2):
    
    return_datas = []
    
    for i in range(len(fake_imgs)):
        img = fake_imgs[i]
        hair_label = label_1[i]
        eyes_label = label_2[i]
        
        return_datas.append((img,hair_label,eyes_label,0))
    
    return return_datas


# In[44]:

total_times = 10000
d_training_times = 1
g_training_times = 1
    
q = 1001

# shuffle(fake_input)
# fake_pakage = data_pakage(fake_input)

d_acc = []
g_acc = []
d_loss = []
g_loss = []

g_model_name = 'g_model.h5'
d_model_name = 'd_model.h5'


for i in range(total_times):

    print('本次是第' + str(q) + '次')

    
    for k in range(d_training_times):
        
###################

        print('making_fake_img')
        g_model = load_model(g_model_name)
        
        fake_imgs = list(g_model.predict(x = ([fake_pakage[0],fake_pakage[1],fake_pakage[2]])))
        fake_imgs = fake_data_split(fake_imgs,fake_pakage[1],fake_pakage[2])
        
        del g_model
        K.clear_session()

###################        

        all_data = fake_imgs + real_input
        shuffle(all_data)
        pakage = data_pakage(all_data)
        
###################
        
        print('training_d_model')
        
        d_model = load_model(d_model_name)
        for layer in d_model.layers[:]:
            layer.trainable = True
        
        d_model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])
        D_TRAIN = d_model.fit(x = ([pakage[0],pakage[1],pakage[2]]), 
                    y = pakage[3], 
                    epochs=1, verbose=1, batch_size = 32)
        d_acc.append(D_TRAIN.history['acc'][0])
        d_loss.append(D_TRAIN.history['loss'][0])
        
        d_model.save(d_model_name)
        del d_model
        K.clear_session()
        
        
    for j in range(g_training_times):
         
        real_label = np.full(len(real_imgs),1)
                
        d_model = load_model(d_model_name)
        g_model = load_model(g_model_name)
        
        for layer in d_model.layers[:]:
            layer.trainable = False
        all_model = Model(inputs = g_model.input,
                    outputs = ([d_model([g_model.output,g_model.input[1],g_model.input[2]])]))
        

        print('training_g_model')
        all_model.compile(optimizer=keras.optimizers.Adam(lr=0.0003),loss='binary_crossentropy',metrics=['accuracy'])
        G_TRAIN = all_model.fit(x = ([fake_pakage[0],fake_pakage[1],fake_pakage[2]]),
                      y = real_label, 
                      epochs=1, verbose=1, batch_size = 32)
        g_acc.append(G_TRAIN.history['acc'][0])
        g_loss.append(G_TRAIN.history['loss'][0])
        
        g_model.save(g_model_name)
        del all_model
        del d_model
        del g_model
        K.clear_session()
        
    if D_TRAIN.history['acc'][0] < 0.55 or G_TRAIN.history['acc'][0] < 0.55 :
        break
    
    if q%5 == 0:
        
        save_(fake_imgs, 5, 'IMAGE_GAN/'+str(q))
    q+=1


# In[45]:

import csv

with open('result_1000_1067_test'+'.csv', mode='w', encoding='utf-8') as write_file:
    writer = csv.writer(write_file, delimiter=',')


    writer.writerow(['epoch','d_acc','d_loss','g_acc','g_loss'])
    for i in range (0,len(g_acc)):
        writer.writerow([str(i),str(d_acc[i]),str(d_loss[i]),str(g_acc[i]),str(g_loss[i])])
        

