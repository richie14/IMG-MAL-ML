#!/usr/bin/env python
# coding: utf-8
import exiftool
import re
import binascii
import os
import zipfile
import pandas as pd
import csv
import os.path
from pathlib import Path
from PIL import Image
from struct import unpack
from collections import defaultdict
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score




#pip install PyExifTool 

# Reading Files

img_folder = Path('/home/seemu/images/clean').rglob('*')
files = [x for x in img_folder]
file_count = len(files)

#extraction of EOI feature
eoi_list = []
eoi = b'\xff\xd9'
for x in files:
    with open (x, 'rb') as f:
        f.seek(-2,os.SEEK_END)
        content = f.read()
        #y= eoi == (content)        
        if content  ==  b'\xff\xd9':
            z=1
        else:
            z=0
        eoi_list.append(z)


#listing all file content
file_list = []
for x in files:
    with open (x, 'rb') as f:
        content = f.read()    
        file_list.append(content)

def count_marker(var1):
    count_marker_list = []
    for i in file_list:
        #hexlify_content = binascii.hexlify(i)
        string_content  = i
        find_marker     = [m.start() for m in re.finditer(var1, string_content)]
        count_marker_list.append((len(find_marker))) 
    return count_marker_list


def size_marker(var1):
    size_marker_list = []
    for i in file_list:
        #hexlify_content = binascii.hexlify(i)
        string_content  = i
        find_marker     = [m.start() for m in re.finditer(var1, string_content)]
        size_marker_list.append((find_marker)) 
    return size_marker_list

#Extracting Total Number of Markers Feature 
all_marker_list = [b'\xff\xe0', b'\xff\xef', b'\xff\xfe', b'\xff\xcc', b'\xff\xde', b'\xff\xc4', b'\xff\xdc', b'\xff\xdb', b'\xff\xdd', b'\xff\xdf', b'\xff\xc8', b'\xff\xf0', b'\xff\xfd', b'\xff\x02', b'\xff\xbf', b'\xff\xd0', b'\xff\xd7',b'\xff\xc0', b'\xff\xda', b'\xff\x01', b'\xff\xd8', b'\xff\xd9']
all_marker_list[5]


# Extract DHT Marker Count
dht_marker_count=count_marker(all_marker_list[5])
dht_marker_count


# Extract DHT Marker Maxsize
dht_marker_maxlist = []
for i in range(len(all_marker_list)):
    dht_marker_maxsize=size_marker(all_marker_list[i])
    dht_marker_maxlist.append(dht_marker_maxsize)
dht_marker_maxlist
final_marker_pos = []
for i in range(0,file_count):
    temp = []
    for j in range(len(dht_marker_maxlist)):
        a = dht_marker_maxlist[j][i]
        temp.append(a)
    final_marker_pos.append(temp)
final_marker_pos[0]

def calc_max_dis(var):#file level	
    res = {all_marker_list[i]: final_marker_pos[var][i] for i in range(len(all_marker_list))}
    new_dic = {}
    for k,v in res.items():
        for x in v:
            new_dic.setdefault(x,[]).append(k)
    for k, v in new_dic.items():
        new_dic[k] = v[0]
    sort_dic = {}
    for i in sorted (new_dic) : 
            sort_dic[i]= new_dic[i]
    sort_dic
    li=[]
    for i,v in sort_dic.items():
        li.append([i,v])

    l1 = []
    for i in range(len(li)-1):
        l1.append([(-li[i][0]+li[i+1][0]),li[i][1]])
    l1.sort()
    l1
    d1={ k[1]: k[0] for k in l1 }
    return d1

dist_max_size = []
for i in range(0,file_count):
    temp1 = calc_max_dis(i)
    dist_max_size.append(temp1)
dist_max_size


# DHT Max Size 
dist_max_size_xffc4 = []
for i in range (0,file_count):
    k2=dist_max_size[i].get(b'\xff\xc4',None)      
    dist_max_size_xffc4.append(k2)
dist_max_size_xffc4


# DQT Max Size 
dist_max_size_xffdb = []
for i in range (0,file_count):
    k2=dist_max_size[i].get(b'\xff\xdb',None)      
    dist_max_size_xffdb.append(k2)
dist_max_size_xffdb

# App1 Max Size 
dist_max_size_xffe0 = []
for i in range (0,file_count):
    k2=dist_max_size[i].get(b'\xff\xe0',None)      
    dist_max_size_xffe0.append(k2)
dist_max_size_xffe0


# App12 Max Size 
dist_max_size_xffef = []
for i in range (0,file_count):
    k2=dist_max_size[i].get(b'\xff\xef',None)      
    dist_max_size_xffef.append(k2)
dist_max_size_xffef

# Com Max Size 
dist_max_size_xfffe = []
for i in range (0,file_count):
    k2=dist_max_size[i].get(b'\xff\xfe',None)      
    dist_max_size_xfffe.append(k2)
dist_max_size_xfffe


# Extract DQT Marker Count
dqt_marker_count = count_marker(all_marker_list[7])
dqt_marker_count


total_marker_list = []
for g in all_marker_list:
    total_marker_list.append(count_marker(g))
total_marker_list
total_marker_sum_list = []
for j in range(len(total_marker_list[0])):
    sum = 0
    for i in range(len(total_marker_list)):
        sum = sum + total_marker_list[i][j]
    total_marker_sum_list.append(sum)
total_marker_sum_list


# File Size Extraction Feature - Percentage of Expected Size
percentage_file_size_list = []
for x in files:
    try:
        im = Image.open(x, mode='r')
    except:
        pass
    filesize=os.path.getsize(x)
    width,height = im.size
    expectedsize = width * height * 3
    percentagesize = round(expectedsize/filesize*100/100,1)
    percentage_file_size_list.append(percentagesize)
percentage_file_size_list


#Static Feature Extraction - Make and Model of JPEG Files
static_meta_data_make_list = []
static_meta_data_model_list = []
for x in files:
    filename_str = str(x)
    with exiftool.ExifTool() as et:
        make  = et.get_tag("EXIF:Make", filename_str)
        if make == "/.*/e":
            MK = 1
        elif(not make):
            MK = 0
        else:
            MK = 2
        static_meta_data_make_list.append(MK)
        model = et.get_tag("EXIF:Model", filename_str)
        if model == "eval(base64_decode('aWYgKGlzc2V0KCRfUE9TVFsienoxIl0pKSB7ZXZhbChzdHJpcHNsYXNoZXMoJF9QT1NUWyJ6ejEiXSkpO30='));":
            MODEL = 1
        elif(not model):
            MODEL = 0
        else:
            MODEL = 2
        static_meta_data_model_list.append(MODEL)
       
static_meta_data_make_list
static_meta_data_model_list

#Getting Ready for data formatting
#print(percentage_file_size_list)
#print(files)

df = pd.DataFrame()
df[""] = eoi_list
df.fillna(value=0, inplace=True)
df["Number of DHT Markers"] = dht_marker_count
df.fillna(value=0, inplace=True)
df["Number of DQT Markers"] = dqt_marker_count
df.fillna(value=0, inplace=True)
df["Total Number of Markers"] = total_marker_sum_list
df.fillna(value=0, inplace=True)
df["Maximum size used in %"] = percentage_file_size_list
df.fillna(value=0, inplace=True)
df["Make of file"] = static_meta_data_make_list 
df.fillna(value=0, inplace=True)
df["Model of file"] = static_meta_data_model_list
df.fillna(value=0, inplace=True)
df["Maximum size used in %"] = percentage_file_size_list
df.fillna(value=0, inplace=True)
#df["Max Marker Size"] = dist_max_size
df["Max Marker Size for DHT(FFC4)"] = dist_max_size_xffc4
df.fillna(value=0, inplace=True)
df["Max Marker Size for DQT FFDB"] = dist_max_size_xffdb
df.fillna(value=0, inplace=True)
df["Max Marker Size for APP1(FFE0)"] = dist_max_size_xffe0
df.fillna(value=0, inplace=True)
df["Max Marker Size for APP12(FFEF)"] = dist_max_size_xffef
df.fillna(value=0, inplace=True)
df["Max Marker Size for COM(FFFE)"] = dist_max_size_xfffe
df.fillna(value=0, inplace=True)
df.rename(columns={0: "Filenames"},inplace=True)
df.to_csv("test.csv", index=False, header=True)
df = pd.read_csv('sample3.csv')

new_input = pd.read_csv('test.csv')

df.head()
x = df.drop('Label',axis = 1)
y = df.Label

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3, random_state=4)

#print("\nGradient Boosting Accuracy:")
gb=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(x_train, y_train)
#print(100 *gb.score(x_test, y_test))

#print("Gradient Boosting Prediction:")
A = gb.predict((np.array(new_input)))
if A == 0:
    print('\033[1m \033[92m \033[4m' +  "File is Benign" + '\033[0m')
else:
#    print("File is Malware")
    print('\033[1m \033[91m \033[4m' +  "File is Malware" + '\033[0m')


