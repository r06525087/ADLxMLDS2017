
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
model = load_model('best_model.h5')


import sys
path_1 = sys.argv[1]
path_2 = sys.argv[2]

time_step = 36

#### 讀取MFCC特徵

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
del process_test_mfcc


#### 讀取fbank特徵

file = open(path_1+'fbank/test.ark')



test_fbank=[]
for line in file:
    test_fbank.append(line)

for i in range (0,len(test_fbank)):
    test_fbank[i]=test_fbank[i].split()
for i in range (0,len(test_fbank)):
    temp=['s',[]]

    for k in range(1,70):
        temp[0]=test_fbank[i][0]
        temp[1].append(test_fbank[i][k])
    test_fbank[i]=temp
for i in range (0,len(test_fbank)):
    for k in range (0,69):
        test_fbank[i][1][k]=float(test_fbank[i][1][k])

for i in range (0,len(test_fbank)):
    test_fbank[i][0]=test_fbank[i][0].replace('_',' ')
for i in range (0,len(test_fbank)):
    test_fbank[i][0]=test_fbank[i][0].split()
for i in range (0,len(test_fbank)):
    test_fbank[i][0]=[test_fbank[i][0][0]+test_fbank[i][0][1],test_fbank[i][0][2]]
for i in range (0,len(test_fbank)):
    test_fbank[i][0][1]=int(test_fbank[i][0][1])

process_test_fbank=[]
temp=[]
for i in range (0,len(test_fbank)):
    if test_fbank[i][0][1]==1:
        process_test_fbank.append(temp)
        
        temp=['name',0]
        temp[0]=test_fbank[i][0][0]
        temp.append(test_fbank[i][1])
    else:
        temp.append(test_fbank[i][1]) 
        
        
process_test_fbank.append(temp)
process_test_fbank.remove([])

#應該會長得像 [ ['名字',int(向量個數),[vector_1],[vector_2]......[vector_n],.. ] , [...] , [...] ...]

process_test_fbank_2=[]
for i in range(0, len(process_test_fbank)):
    temp = [[],[]]
    temp[0].append(process_test_fbank[i][0])
    temp[0].append(len(process_test_fbank[i])-2)
    for k in range(2,len(process_test_fbank[i])):
        temp[1].append(process_test_fbank[i][k])
    
    process_test_fbank_2.append(temp)
del process_test_fbank


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

process_test_mfcc_2 = seperate_test_data(process_test_mfcc_2,time_step,np.zeros(39))
process_test_fbank_2 = seperate_test_data(process_test_fbank_2,time_step,np.zeros(69))

def evaluate_all(input_list_1,input_list_2):
    return_list = []
    
    for i in range(0,len(input_list_1[1])):
        one_person_data = []
        
        one_person_data.append(input_list_1[0][i][0]) #小list的第一筆是第i筆資料的人名

        i_predict = model.predict(x=([np.array(input_list_1[1][i]),
                                      np.array(input_list_2[1][i])])) #將第i筆資料的data取出來預測
        i_predict = list(i_predict)
        
        predict_list = [] #用來裝預測完的東西
        
        for m in range(0,len(i_predict)):
            for n in range(0,len(i_predict[0])):
                predict_list.append(list(i_predict[m][n]))
            
        
        phone_number=[]                        #用來裝一串數字
        
        for k in range(0,input_list_1[0][i][1]): #k是實際有的PHONE數量 所以只做K次
            
            idx = predict_list[k].index(max(predict_list[k]))
            
            phone_number.append(idx)
            
        
        one_person_data.append(phone_number)
        
        return_list.append(one_person_data)
    
    return return_list

result=evaluate_all(process_test_mfcc_2,process_test_fbank_2)


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
        
        
##################   TRAINING  CODE   ##################

# import csv

# all_data =[]

# with open('train.csv', mode='r', encoding='utf-8') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         row[0] = row[0].replace('_',' ') #把底線用空白代替
#         row[0] = row[0].split()          #讀到空白就分開
#         all_data.append(row)
#     #    print(row)
# all_data[0][0]=['maeb0', 'si1411', '1'] 

# for i in range (0,len(all_data)):
#     all_data[i][0]=[all_data[i][0][0]+all_data[i][0][1],all_data[i][0][2]]

# process_data = [ [['maeb0si1411', '1'], 'sil'] ] 

# k=0

# for i in range (1,len(all_data)):
#     if all_data[i][0][0] == process_data[k][0][0]:
#         process_data[k].append(all_data[i][1])
#     else:
#         process_data.append(all_data[i])
#         k+=1

# for i in range (0,len(process_data)):
#     process_data[i][0][1]=len(process_data[i])-1

# sentence_lenth = []
# for i in range (0,len(process_data)):
#     sentence_lenth.append(process_data[i][0][1])

# #process_data是目前的 [ [ ['名字',音節長度],'phone1','phone2',... ] , [ ] ....] 


# process_data_2 = []
# for i in range (0,len(process_data)):
#     temp = [[],[]]
#     temp[0]=(process_data[i][0])
#     for k in range(1,process_data[i][0][1]+1):
#         temp[1].append(process_data[i][k])
#     process_data_2.append(temp)
# #應該會長得像 [ ['名字', 音節長度 ] , ['phone1','phone2',... ] ]

# phone_change = {'ao':'aa', 'ax':'ah',  'cl':'sil','el':'l',
#                 'en':'n' , 'epi':'sil','ix':'ih', 'vcl':'sil','zh':'sh'}

# for i in range (0,len(process_data_2)):
#     process_data_2[i][1] = [phone_change[x] if x in phone_change else x for x in process_data_2[i][1]]
    

# label_vector={
# 'aa':0,
# 'ae':1,
# 'ah':2,

# 'aw':3,

# 'ay':4,
# 'b':5,
# 'ch':6,

# 'd':7,
# 'dh':8,
# 'dx':9,
# 'eh':10,

# 'er':11,
# 'ey':12,
# 'f':13,
# 'g':14,
# 'hh':15,
# 'ih':16,

# 'iy':17,
# 'jh':18,
# 'k':19,
# 'l':20,
# 'm':21,
# 'n':22,
# 'ng':23,
# 'ow':24,
# 'oy':25,
# 'p':26,
# 'r':27,
# 's':28,
# 'sh':29,
# 'sil':30,
# 't':31,
# 'th':32,
# 'uh':33,
# 'uw':34,
# 'v':35,

# 'w':36,
# 'y':37,
# 'z':38,
# }
# for i in range(0,len(process_data_2)):
    
#     process_data_2[i][1] = [label_vector[x] if x in label_vector else x for x in process_data_2[i][1]] 

# import numpy as np

# for i in range(0,len(process_data_2)):
#     for k in range (0,process_data_2[i][0][1]):
#         num = process_data_2[i][1][k]
#         process_data_2[i][1][k]=np.zeros(40) #補到40維 多一個是等等用來補0的
#         process_data_2[i][1][k][num]=1

        
# file = open('train_mfcc.txt')

# mfcc=[]
# for line in file:
#     mfcc.append(line)

# for i in range (0,len(mfcc)):
#     mfcc[i]=mfcc[i].split()
# for i in range (0,len(mfcc)):
#     temp=['s',[]]

#     for k in range(1,40):
#         temp[0]=mfcc[i][0]
#         temp[1].append(mfcc[i][k])
#     mfcc[i]=temp
# for i in range (0,len(mfcc)):
#     for k in range (0,39):
#         mfcc[i][1][k]=float(mfcc[i][1][k])

# for i in range (0,len(mfcc)):
#     mfcc[i][0]=mfcc[i][0].replace('_',' ')
# for i in range (0,len(mfcc)):
#     mfcc[i][0]=mfcc[i][0].split()
# for i in range (0,len(mfcc)):
#     mfcc[i][0]=[mfcc[i][0][0]+mfcc[i][0][1],mfcc[i][0][2]]
# for i in range (0,len(mfcc)):
#     mfcc[i][0][1]=int(mfcc[i][0][1])

# process_mfcc=[]
# temp=[]
# for i in range (0,len(mfcc)):
#     if mfcc[i][0][1]==1:
#         process_mfcc.append(temp)
        
#         temp=['name',0]
#         temp[0]=mfcc[i][0][0]
#         temp.append(mfcc[i][1])
#     else:
#         temp.append(mfcc[i][1]) 
        
        
# process_mfcc.append(temp)
# process_mfcc.remove([])

# #應該會長得像 [ ['名字',int(向量個數),[vector_1],[vector_2]......[vector_n],.. ] , [...] , [...] ...]

# process_mfcc_2=[]
# for i in range(0, len(process_mfcc)):
#     temp = [[],[]]
#     temp[0].append(process_mfcc[i][0])
#     temp[0].append(len(process_mfcc[i])-2)
#     for k in range(2,len(process_mfcc[i])):
#         temp[1].append(process_mfcc[i][k])
    
#     process_mfcc_2.append(temp)
    
# del process_mfcc
# #變成 [['名字',向量數],[ 好多個向量list在這   ]]


# for i in range (0,len(process_data_2)):
#     for k in range(0,len(process_mfcc_2)):
#         if process_mfcc_2[k][0][0] == process_data_2[i][0][0]:
#             temp = []
#             temp = process_mfcc_2[k]
#             process_mfcc_2[k] = process_mfcc_2[i]
#             process_mfcc_2[i] = temp

# file = open('train_fbank.txt')

# fbank=[]
# for line in file:
#     fbank.append(line)

# for i in range (0,len(fbank)):
#     fbank[i]=fbank[i].split()
# for i in range (0,len(fbank)):
#     temp=['s',[]]

#     for k in range(1,70):
#         temp[0]=fbank[i][0]
#         temp[1].append(fbank[i][k])
#     fbank[i]=temp
# for i in range (0,len(fbank)):
#     for k in range (0,69):
#         fbank[i][1][k]=float(fbank[i][1][k])

# for i in range (0,len(fbank)):
#     fbank[i][0]=fbank[i][0].replace('_',' ')
# for i in range (0,len(fbank)):
#     fbank[i][0]=fbank[i][0].split()
# for i in range (0,len(fbank)):
#     fbank[i][0]=[fbank[i][0][0]+fbank[i][0][1],fbank[i][0][2]]
# for i in range (0,len(fbank)):
#     fbank[i][0][1]=int(fbank[i][0][1])

# process_fbank=[]
# temp=[]
# for i in range (0,len(fbank)):
#     if fbank[i][0][1]==1:
#         process_fbank.append(temp)
        
#         temp=['name',0]
#         temp[0]=fbank[i][0][0]
#         temp.append(fbank[i][1])
#     else:
#         temp.append(fbank[i][1]) 
        
        
# process_fbank.append(temp)
# process_fbank.remove([])

# #應該會長得像 [ ['名字',int(向量個數),[vector_1],[vector_2]......[vector_n],.. ] , [...] , [...] ...]

# process_fbank_2=[]
# for i in range(0, len(process_fbank)):
#     temp = [[],[]]
#     temp[0].append(process_fbank[i][0])
#     temp[0].append(len(process_fbank[i])-2)
#     for k in range(2,len(process_fbank[i])):
#         temp[1].append(process_fbank[i][k])
    
#     process_fbank_2.append(temp)
# del process_fbank
# #變成 [['名字',向量數],[ 好多個向量list在這   ]]

# for i in range (0,len(process_fbank_2)):
#     for k in range(0,len(process_fbank_2)):
#         if process_fbank_2[k][0][0] == process_data_2[i][0][0]:
#             temp = []
#             temp = process_fbank_2[k]
#             process_fbank_2[k] = process_fbank_2[i]
#             process_fbank_2[i] = temp
#             count +=1
# def seperate_data(input_list,num,fill_thing):

#     seperated_data = []
#     data_part = []  # [ ] = 裡面是每一筆都是不同句的 data list

#     for i in range(0,len(input_list)):              #進來的第i筆資料先取出第1項
#         data_part.append(input_list[i][1])
    
#     for i in range (0,len(data_part)):              #對於DATA PART的第i筆資料
#         prepare_num = int(len(data_part[i])/num)+1  #prepare_num = 替每一筆資料要準備的LIST數
        
#         for k in range(0,prepare_num*num-len(data_part[i])): #先幫每筆資料補滿0
#             data_part[i].append(fill_thing)                  #從0到 ~ (LIST數*num-原本長度) 為要補東西的數量
        
#         for j in range(0,prepare_num):              #製作 prepare_num數量的LIST
#             ok_data=[]
            
#             ok_data = data_part[i][num*j:num*(j+1)]
            
#             seperated_data.append(ok_data)
    
#     return seperated_data

# ###
# time_step = 36

# ###製作空缺的標籤

# no_voice_label = np.zeros(40)
# no_voice_label[39]=1
# no_voice_label = np.array(no_voice_label)
# print(no_voice_label)

# ###

# input_x_1 = seperate_data(process_mfcc_2,time_step,np.zeros(39))  
# input_x_2 = seperate_data(process_fbank_2,time_step,np.zeros(69))
# input_y = seperate_data(process_data_2,time_step,no_voice_label)
# ###

# input_x_1=np.array(input_x_1)
# input_x_2=np.array(input_x_2)
# input_y=np.array(input_y)

# print(input_x_1.shape,input_x_2.shape,input_y.shape)


# ####
# input_mfcc = Input(shape=(time_step,39))
# mfcc_BN = BatchNormalization()(input_mfcc)
# mfcc_conv = Conv1D(filters=128,
#                    kernel_size=7,
#                    padding='causal',
#                    strides=1)(mfcc_BN)
# mfcc_activation_1 = LeakyReLU()(mfcc_conv)
# mfcc_drop = Dropout(0.25)(mfcc_activation_1)


# ####
# input_fbank = Input(shape=(time_step,69))
# fbank_BN = BatchNormalization()(input_fbank)
# fbank_conv = Conv1D(filters=128,
#                    kernel_size=7,
#                    padding='causal',
#                    strides=1)(fbank_BN)

# fbank_activation_1 = LeakyReLU()(fbank_conv)
# fbank_drop = Dropout(0.25)(fbank_activation_1)

# ####

# merge = Add()([mfcc_drop,fbank_drop])

# All_layer = BatchNormalization()(merge)

# All_LSTM_1 = Bidirectional(LSTM(units=128,return_sequences=True))(All_layer)
# Activation_1 = LeakyReLU()(All_LSTM_1)
# All_Drop_1 = Dropout(0.25)(Activation_1)

# All_LSTM_2 = Bidirectional(LSTM(units=256,return_sequences=True))(All_Drop_1)
# Activation_2 = LeakyReLU()(All_LSTM_2)
# All_Drop_2 = Dropout(0.25)(Activation_2)


# All_LSTM_4 = Bidirectional(LSTM(units=256,return_sequences=True))(All_Drop_2)
# Activation_4 = LeakyReLU()(All_LSTM_4)
# All_Drop_4 = Dropout(0.25)(Activation_4)



# Time = TimeDistributed(Dense(units=40))(All_Drop_4)


# result_layer = Activation('softmax')(Time)


# model = Model(inputs=[input_mfcc, input_fbank], outputs=result_layer)
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# model.summary()

# model.fit( x=([input_x_1,input_x_2]),
#            y= input_y ,
#            batch_size=150,
#            epochs=6,
#            validation_split=0)
    

