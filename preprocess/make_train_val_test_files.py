# -*- coding: utf-8 -*-
"""
Created on Wed May  5 19:45:05 2021

@author: Prthamesh
"""

path = r'F:\EDUCATION\MS\3_Spring_2021\Natural_Language_Processing\Project\Data'

f_text = open(path + r'\text.txt','r')
f_summary = open(path + r'\summary.txt','r')
f_label = open(path + r'\label.txt','r')

val_size = 1000
test_size = 1000

# val
f_text_val = open(path + r'\train_val_test_data\val\text_val.txt','w')
f_summary_val = open(path + r'\train_val_test_data\val\summary_val.txt','w')
f_label_val = open(path + r'\train_val_test_data\val\label_val.txt','w')

for i,line in enumerate(f_text):
    f_text_val.write(line)
    if i == (val_size-1):break

for i,line in enumerate(f_summary):
    f_summary_val.write(line)
    if i == (val_size-1):break

for i,line in enumerate(f_label):
    f_label_val.write(line)
    if i == (val_size-1):break
    
f_text_val.close()
f_summary_val.close()
f_label_val.close()

# test
f_text_test = open(path + r'\train_val_test_data\test\text_test.txt','w')
f_summary_test = open(path + r'\train_val_test_data\test\summary_test.txt','w')
f_label_test = open(path + r'\train_val_test_data\test\label_test.txt','w')

for i,line in enumerate(f_text):
    f_text_test.write(line)
    if i == (test_size-1):break

for i,line in enumerate(f_summary):
    f_summary_test.write(line)
    if i == (test_size-1):break

for i,line in enumerate(f_label):
    f_label_test.write(line)
    if i == (test_size-1):break

f_text_test.close()
f_summary_test.close()
f_label_test.close()

# train

f_text_train = open(path + r'\train_val_test_data\train\text_train.txt','w')
f_summary_train = open(path + r'\train_val_test_data\train\summary_train.txt','w')
f_label_train = open(path + r'\train_val_test_data\train\label_train.txt','w')

for i,line in enumerate(f_text):
    f_text_train.write(line)

for i,line in enumerate(f_summary):
    f_summary_train.write(line)

for i,line in enumerate(f_label):
    f_label_train.write(line)

f_text_train.close()
f_summary_train.close()
f_label_train.close()

f_text.close()
f_summary.close()
f_label.close()