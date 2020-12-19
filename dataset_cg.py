# -*- coding: UTF-8 -*-
import os

#获得文件夹下文件名列表
path=r"./speech_number0-9_dataset/"
file_list=os.listdir(path)

#选择要重命名的文件夹路径
os.chdir(path)

#将文件名中的第一个'_'用'.'替代
count = 0
for file in file_list:
  count = count +1
  print(count)
  os.rename(file,file.replace('_','.',1))

