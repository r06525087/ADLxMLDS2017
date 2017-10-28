
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import keras
from keras.layers import Concatenate,Masking,Add,concatenate, UpSampling1D,MaxPooling1D,Input,Dense,LSTM,TimeDistributed,Activation,Dropout,BatchNormalization,Conv1D,Bidirectional
from keras.models import Model, Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.core import RepeatVector
from keras.layers import convolutional

from keras.models import load_model
model = load_model('cnn_model.h5')


import sys
path_1 = sys.argv[1]
path_2 = sys.argv[2]

input_time_step = 36

file = open(path_1+'mfcc/test.ark')




test_mfcc=[]
for line in file:
    test_mfcc.append(line)

for i in range (0,len(test_mfcc)):
    test_mfcc[i]=test_mfcc[i].split()
for i in range (0,len(test_mfcc)):
    temp=['s',[]]

    for k in range(1,40):
        temp[0]=test_mfcc[i][0]
        temp[1].append(test_mfcc[i][k])
    test_mfcc[i]=temp
for i in range (0,len(test_mfcc)):
    for k in range (0,39):
        test_mfcc[i][1][k]=float(test_mfcc[i][1][k])

for i in range (0,len(test_mfcc)):
    test_mfcc[i][0]=test_mfcc[i][0].replace('_',' ')
for i in range (0,len(test_mfcc)):
    test_mfcc[i][0]=test_mfcc[i][0].split()
for i in range (0,len(test_mfcc)):
    test_mfcc[i][0]=[test_mfcc[i][0][0]+test_mfcc[i][0][1],test_mfcc[i][0][2]]
for i in range (0,len(test_mfcc)):
    test_mfcc[i][0][1]=int(test_mfcc[i][0][1])

process_test_mfcc=[]
temp=[]
for i in range (0,len(test_mfcc)):
    if test_mfcc[i][0][1]==1:
        process_test_mfcc.append(temp)
        
        temp=['name',0]
        temp[0]=test_mfcc[i][0][0]
        temp.append(test_mfcc[i][1])
    else:
        temp.append(test_mfcc[i][1]) 
        
        
process_test_mfcc.append(temp)
process_test_mfcc.remove([])

#應該會長得像 [ ['名字',int(向量個數),[vector_1],[vector_2]......[vector_n],.. ] , [...] , [...] ...]

process_test_mfcc_2=[]
for i in range(0, len(process_test_mfcc)):
    temp = [[],[]]
    temp[0].append(process_test_mfcc[i][0])
    temp[0].append(len(process_test_mfcc[i])-2)
    for k in range(2,len(process_test_mfcc[i])):
        temp[1].append(process_test_mfcc[i][k])
    
    process_test_mfcc_2.append(temp)
#變成 [['名字',向量數],[ 好多個向量list在這   ]]

# 寫函數 : input = 標準資料格式 
#         output = 1個list包含n個人的SEQ, 每個人的SEQ由seq_len倍數的切開, 不足補0向量

def seperate_test_data(input_data_2,seq_len,fill_thing):
    
    return_list = []
    
    
    name_list=[]
    
    for i in range(0,len(input_data_2)):
        name_list.append(input_data_2[i][0])
    
    

    data_list = []
    
    for i in range(0,len(input_data_2)):
        
        one_sentence = []
        phone_num = (int( len(input_data_2[i][1])/seq_len ) +1)*seq_len #計算每個句子補完後該有的長度
        
        for k in range(0,phone_num-len(input_data_2[i][1])):
            input_data_2[i][1].append(fill_thing)  #長度不夠的部分由fill_thing補完
        
        for j in range(0,int(phone_num/seq_len)):
            part_list = []                         #準備整串PHONE/SEQ_LEN數量的LIST
            part_list = input_data_2[i][1][seq_len*j:seq_len*(j+1)]
        
            one_sentence.append(part_list)
        
        data_list.append(one_sentence)
    
    return_list.append(name_list)
    return_list.append(data_list)
    
    return return_list
        
        

process_test_mfcc_2 = seperate_test_data(process_test_mfcc_2,input_time_step,np.zeros(39))

# input = 經過切開及fill的資料
# output = 一個大list 每個元素為 [[id],[預測出的結果還原回1~39的數字]]

def evaluate_all(input_list):
    return_list = []
    
    for i in range(0,len(input_list[1])):
        one_person_data = []
        
        one_person_data.append(input_list[0][i][0]) #小list的第一筆是第i筆資料的人名

        i_predict = model.predict(x=np.array(input_list[1][i])) #將第i筆資料的data取出來預測
        i_predict = list(i_predict)
        
        predict_list = [] #用來裝預測完的東西
        
        for m in range(0,len(i_predict)):
            for n in range(0,len(i_predict[0])):
                predict_list.append(list(i_predict[m][n]))
            
        
        phone_number=[]                        #用來裝一串數字
        
        for k in range(0,input_list[0][i][1]): #k是實際有的PHONE數量 所以只做K次
            
            idx = predict_list[k].index(max(predict_list[k]))
            
            phone_number.append(idx)
            
        
        one_person_data.append(phone_number)
        
        return_list.append(one_person_data)
    
    return return_list

result=evaluate_all(process_test_mfcc_2)

for i in range(0,len(result)):
    for k in range(0,len(result[i][1])):
        if result[i][1][k]==39:

            result[i][1][k]=30



def delete_sil_continue(input_result):
    for i in range(len(input_result)):

        data_list=[]
        data_list.append(input_result[i][1][0])
    
        for k in range(0,len(input_result[i][1])):
            if input_result[i][1][k]!=data_list[len(data_list)-1]:
                data_list.append(input_result[i][1][k])
        
        if data_list[0]==30:
            del(data_list[0])
        if data_list[-1]==30:
            del(data_list[-1])
        
        input_result[i][1]=data_list

delete_sil_continue(result)


num_to_phone={0:'aa',  1:'ae',  2:'ah',  3:'aw',  4:'ay',  5:'b',  6:'ch',  7:'d',  8:'dh',  9:'dx',  
              10:'eh',11:'er', 12:'ey',  13:'f',  14:'g',15:'hh', 16:'ih',17:'iy', 18:'jh',  19:'k',  
               20:'l', 21:'m',  22:'n', 23:'ng', 24:'ow',25:'oy',  26:'p', 27:'r',  28:'s', 29:'sh',
             30:'sil', 31:'t' ,32:'th', 33:'uh', 34:'uw', 35:'v',  36:'w', 37:'y',  38:'z'}

final_diction={'aa':'a', 'ae':'b', 'ah':'c', 'aw':'e', 'ay':'g', 'b':'h', 'ch':'i', 'd':'k', 'dh':'l', 'dx':'m',
               'eh':'n', 'er':'r', 'ey':'s', 'f':'t',  'g':'u',  'hh':'v','ih':'w', 'iy':'y','jh':'z', 'k':'A',
               'l' :'B', 'm':'C',  'n':'D',  'ng':'E', 'ow':'F', 'oy':'G','p':'H',  'r':'I', 's':'J',  'sh':'K',
               'sil':'L','t':'M',  'th':'N', 'uh':'O', 'uw':'P', 'v':'Q', 'w':'S',  'y':'T', 'z':'U'}




def turn_num_to_phone(input_result):
    final_result=[]
    
    for i in range(0,len(input_result)):
        the_result=[]
       
        name = input_result[i][0]
        name = name[0:5]+'_'+name[5:]
        
        the_result.append(name) #名字

       
        input_result[i][1]=[num_to_phone[x] if x in num_to_phone else x for x in input_result[i][1]] 
        input_result[i][1]=[final_diction[x] if x in final_diction else x for x in input_result[i][1]]
        
        first_phone=input_result[i][1][0]
        
        for k in range(1,len(input_result[i][1])):
            first_phone = first_phone+input_result[i][1][k]
        
        the_result.append(first_phone)
    
        final_result.append(the_result)
    
    return final_result

OK_result = turn_num_to_phone(result) 



import csv

with open(path_2+'.csv', mode='w', encoding='utf-8') as write_file:
    writer = csv.writer(write_file, delimiter=',')


    writer.writerow(['id','phone_sequence'])
    for i in range (0,len(OK_result)):
        writer.writerow(OK_result[i])



        
###  資料前處理跟 BEST 裡面都一樣  模型不同~~      
        
# model = Sequential()

# model.add(BatchNormalization(input_shape=(input_time_step,39)))

# model.add(Conv1D(filters=64,
#                  kernel_size=7,
#                  padding='same',
#                  strides=1
#                  ))
# model.add(LeakyReLU())
# model.add(Dropout(0.2))

# model.add(Conv1D(filters=128,
#                  kernel_size=7,
#                  padding='same',
#                  strides=1
#                  ))
# model.add(LeakyReLU())
# model.add(Dropout(0.2))

# model.add(Conv1D(filters=128,
#                  kernel_size=7,
#                  padding='same',
#                  strides=1
#                  ))
# model.add(LeakyReLU())
# model.add(Dropout(0.2))


# model.add(BatchNormalization())

# model.add(Bidirectional(LSTM(units=128,return_sequences=True)))
# model.add(LeakyReLU())
# model.add(Dropout(0.2))
          
# model.add(BatchNormalization())

# model.add(Bidirectional(LSTM(units=256,return_sequences=True)))
# model.add(LeakyReLU())
# model.add(Dropout(0.2))
# model.add(BatchNormalization())


# model.add(TimeDistributed(Dense(40)))
# model.add(LeakyReLU())
# model.add(Dropout(0.2))




# model.add(BatchNormalization())
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.summary()

