#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#pip install PyExifTool 


# In[3]:


# Reading Files

#img_folder = Path('/home/seemu/capstonedata/malicious').rglob('*')
files = Path('/home/richa/Documents/CapstonProject/demo.jpg').rglob('*')
files
file_count = len(files)


# In[4]:


#extraction of EOI feature
eoi_list = []
eoi = b'\xff\xd9'
for x in files:
    with open (x, 'rb') as f:
        f.seek(-2,os.SEEK_END)
        content = f.read()
        y= eoi == (content)        
        eoi_list.append(y)
eoi_list


# In[5]:


#listing all file content
file_list = []
for x in files:
    with open (x, 'rb') as f:
        content = f.read()    
        file_list.append(content)


# In[6]:


def count_marker(var1):
    count_marker_list = []
    for i in file_list:
        #hexlify_content = binascii.hexlify(i)
        string_content  = i
        find_marker     = [m.start() for m in re.finditer(var1, string_content)]
        count_marker_list.append((len(find_marker))) 
    return count_marker_list


# In[7]:


def size_marker(var1):
    size_marker_list = []
    for i in file_list:
        #hexlify_content = binascii.hexlify(i)
        string_content  = i
        find_marker     = [m.start() for m in re.finditer(var1, string_content)]
        size_marker_list.append((find_marker)) 
    return size_marker_list


# In[8]:


#Extracting Total Number of Markers Feature 
all_marker_list = [b'\xff\xe0', b'\xff\xef', b'\xff\xfe', b'\xff\xcc', b'\xff\xde', b'\xff\xc4', b'\xff\xdc', b'\xff\xdb', b'\xff\xdd', b'\xff\xdf', b'\xff\xc8', b'\xff\xf0', b'\xff\xfd', b'\xff\x02', b'\xff\xbf', b'\xff\xd0', b'\xff\xd7',b'\xff\xc0', b'\xff\xda', b'\xff\x01', b'\xff\xd8', b'\xff\xd9']
all_marker_list[5]


# In[9]:


# Extract DHT Marker Count
dht_marker_count=count_marker(all_marker_list[5])
dht_marker_count


# In[10]:


# Extract DHT Marker Maxsize
dht_marker_maxlist = []
for i in range(len(all_marker_list)):
    dht_marker_maxsize=size_marker(all_marker_list[i])
    dht_marker_maxlist.append(dht_marker_maxsize)
dht_marker_maxlist


# In[11]:


final_marker_pos = []
for i in range(0,file_count):
    temp = []
    for j in range(len(dht_marker_maxlist)):
        a = dht_marker_maxlist[j][i]
        temp.append(a)
    final_marker_pos.append(temp)
final_marker_pos[0]


# In[12]:


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
    #print(l1)
    l1.sort()
    l1
    d1={ k[1]: k[0] for k in l1 }
    return d1


# In[13]:


dist_max_size = []
for i in range(0,file_count):
    temp1 = calc_max_dis(i)
    dist_max_size.append(temp1)
dist_max_size


# In[27]:


# DHT Max Size 
dist_max_size_xffc4 = []
for i in range (0,file_count):
    k2=dist_max_size[i].get(b'\xff\xc4',None)      
    dist_max_size_xffc4.append(k2)
dist_max_size_xffc4


# In[28]:


# DQT Max Size 
dist_max_size_xffdb = []
for i in range (0,file_count):
    k2=dist_max_size[i].get(b'\xff\xdb',None)      
    dist_max_size_xffdb.append(k2)
dist_max_size_xffdb


# In[32]:


# App1 Max Size 
dist_max_size_xffe0 = []
for i in range (0,file_count):
    k2=dist_max_size[i].get(b'\xff\xe0',None)      
    dist_max_size_xffe0.append(k2)
dist_max_size_xffe0


# In[37]:


# App12 Max Size 
dist_max_size_xffef = []
for i in range (0,file_count):
    k2=dist_max_size[i].get(b'\xff\xef',None)      
    dist_max_size_xffef.append(k2)
dist_max_size_xffef


# In[38]:


# Com Max Size 
dist_max_size_xfffe = []
for i in range (0,file_count):
    k2=dist_max_size[i].get(b'\xff\xfe',None)      
    dist_max_size_xfffe.append(k2)
dist_max_size_xfffe


# In[39]:


# Extract DQT Marker Count
dqt_marker_count = count_marker(all_marker_list[7])
dqt_marker_count


# In[40]:


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


# In[41]:


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



# In[42]:


#Getting Ready for data formatting
print(percentage_file_size_list)
print(files)

df = pd.DataFrame(files)
df["EOI"] = eoi_list
df["NoDHTMrkrs"] = dht_marker_count
df["NoDQTMrkrs"] = dqt_marker_count
df["NoOfMarkers"] = total_marker_sum_list
df["MaxSize%"] = percentage_file_size_list
df["Make of file"] = static_meta_data_make_list 
df["Model of file"] = static_meta_data_model_list
df["MaxSize%"] = percentage_file_size_list
#df["Max Marker Size"] = dist_max_size
df["DHT(FFC4)"] = dist_max_size_xffc4
df["DQT FFDB"] = dist_max_size_xffdb
df["APP1(FFE0)"] = dist_max_size_xffe0
df["APP12(FFEF)"] = dist_max_size_xffef
df["COM(FFFE)"] = dist_max_size_xfffe
df.to_csv("jpeg_featue_demo.csv", index=False)
df



