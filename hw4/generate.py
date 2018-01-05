
# coding: utf-8

# In[9]:

import numpy as np
import keras
from keras.layers import Concatenate,Add,Input,Dense,Activation,Conv2DTranspose,Reshape,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential, load_model
import random as rnd
from PIL import Image
import math


# In[ ]:

import sys
path_1 = sys.argv[1]


# In[2]:

def noise_maker(m, n):
    noise = np.random.uniform(-1,1.,size=[m,n])
    return noise

def result_check(hair_style, eyes_style, noise, model):
    nums = len(noise)
    hair_vector = hair_condition[hair_style]
    eyes_vector = eyes_condition[eyes_style]
    hair_input=[]
    eyes_input=[]
    for i in range(nums):
        hair_input.append(hair_vector)
        eyes_input.append(eyes_vector)
    
    hair_input = np.array(hair_input)
    eyes_input = np.array(eyes_input)
    
    results = model.predict(x=([noise,hair_input,eyes_input]))
    
    results = results*255
    results = results.astype('uint8')
    
    pictures = []
    
    for k in range(nums):
        pillow_image = Image.fromarray(results[k])
        pillow_image = pillow_image.resize( (64, 64), Image.BILINEAR )
        pictures.append(pillow_image)
    return pictures

def input_condition_maker(input_string):
    input_string = input_string.split(' ')
    if len(input_string) == 4:
        hair_string = input_string[0]+' '+input_string[1]
        eyes_string = input_string[2]+' '+input_string[3]
    
    if len(input_string) == 2:
        
        if 'hair' in input_string:
            hair_string = input_string[0]+' '+input_string[1]
            eyes_string = rnd.sample(hw_eyes,1)[0]
        if 'eyes' in input_string:
            eyes_string = input_string[0]+' '+input_string[1]
            hair_string = rnd.sample(hw_hair,1)[0]
    
    return (hair_string,eyes_string)

def open_file(file_path):
    with open(file_path) as f:
        content = f.readlines()
    
    conditions = []
    
    for i in range(len(content)):
        
        if i==len(content)-1:
            condition = content[i]
            condition = condition[2:]
            conditions.append(condition)
        
        else:
            condition = content[i]
            condition = condition[2:-1]
            conditions.append(condition)
    
    return conditions


# In[3]:

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
    
noise_1 = np.load('pre_noise/noise_1.npy')
noise_2 = np.load('pre_noise/noise_2.npy')
noise_3 = np.load('pre_noise/noise_3.npy')

all_noise = [noise_1,noise_2,noise_3]


# In[4]:

model = load_model('trained_model.h5')


# In[5]:

conditions = open_file(path_1)
print(conditions)

new_conditions = []
for i in range(len(conditions)):
    condition = input_condition_maker(conditions[i])
    new_conditions.append(condition)
print(new_conditions)
    


# In[8]:

for k in range(len(new_conditions)):

    result_test = result_check(hair_style = new_conditions[k][0], 
                               eyes_style = new_conditions[k][1], 
                               noise = all_noise[k], 
                               model = model)

    for i in range(len(result_test)):
        img = result_test[i]
        img_name = 'samples/'+'sample_'+str(k+1)+'_'+str(i+1)+'.jpg'
        img.save(img_name)

