#!/usr/bin/env python
# coding: utf-8

# <b>libraries</b>

# In[1]:

#pip3 install py7zr
#import zipfile
import json
import pandas as pd
import py7zr
from collections import defaultdict


# <b>data inputs, reading jsons from zip file</b>

# In[2]:


d = None  
data = None
temp_list = []
final_data = []
filenames = []
#df = pd.DataFrame()
with py7zr.SevenZipFile("batch1.7z", "r") as z:
    for filename in z.namelist():  
        filenames.append(filename)  
        with z.open(filename) as f:  
            data = f.read()  
            d = json.loads(data)
            temp_list=d
            final_data.append(temp_list)
filenames_dict= dict(enumerate(filenames))


# <b>relevant extraction functions</b>

# In[3]:


def create_dict_field(final_data,field_name):
    field_dict ={}
    for i in range(len(final_data)):
        k1=final_data[i].get("behavior",-99)
        if(k1==-99):
            field_dict[i]=0
        else:
            k2=final_data[i]["behavior"].get("summary",-99)
            if(k2==-99):
                field_dict[i]=0
            else:
                k3=final_data[i]["behavior"]["summary"].get(field_name,-99)
                if(k3==-99):
                    field_dict[i]=0
                else:
                    field_len=len(final_data[i]["behavior"]["summary"][field_name])
                    field_dict[i]=field_len
    return field_dict

def sing_count(final_data):
    sign_dict ={}
    for i in range(len(final_data)):
        k1=final_data[i].get("signatures",-99)
        if(k1==-99):
            sign_dict[i]=0
        else:
            var = len(final_data[i]["signatures"])
            sign_dict[i]=var            
    return sign_dict  


# <b>report initialization</b>

# In[4]:


report_df = pd.DataFrame.from_dict(filenames_dict, orient='index')


# <b>function calls and calculations</b>

# In[5]:


file_deleted = create_dict_field(final_data,"file_deleted")
report_df["file_deleted"] = pd.DataFrame.from_dict(file_deleted, orient='index')
file_created = create_dict_field(final_data,"file_created")
report_df["file_created"] = pd.DataFrame.from_dict(file_created, orient='index')
file_written = create_dict_field(final_data,"file_written")
report_df["file_written"] = pd.DataFrame.from_dict(file_written, orient='index')
directory_created = create_dict_field(final_data,"directory_created")
report_df["directory_created"] = pd.DataFrame.from_dict(directory_created, orient='index')
dll_loaded = create_dict_field(final_data,"dll_loaded")
report_df["dll_loaded"] = pd.DataFrame.from_dict(dll_loaded, orient='index')
regkey_opened = create_dict_field(final_data,"regkey_opened")
report_df["regkey_opened"] = pd.DataFrame.from_dict(regkey_opened, orient='index')
signatures = sing_count(final_data)
report_df["signatures"] = pd.DataFrame.from_dict(signatures, orient='index')


# <b>export to relevant data store (csv)</b>

# In[6]:


report_df.rename(columns={0: "filenames"},inplace=True)
report_df.to_csv("report_summary.csv",index=False)

