# -*- coding: utf-8 -*-
"""
Created on Wed May  5 19:31:36 2021

@author: Prthamesh
"""

path = r'F:\EDUCATION\MS\3_Spring_2021\Natural_Language_Processing\Project\Data'

f_text = open(path + r'\text.txt','w')
f_summary = open(path + r'\summary.txt','w')
f_label = open(path + r'\label.txt','w')

f = open(r'F:\EDUCATION\MS\3_Spring_2021\Natural_Language_Processing\Project\Data\Toys_&_Games\Toys_&_Games.txt')

text_count = summary_count = label_count = 0

for line in f:
    if line.startswith('review/score'):
        l = line.split(':')
        f_label.write(l[1].strip()+'\n')
        label_count += 1
    elif line.startswith('review/summary'):
        l = line.split(':')
        f_summary.write(l[1].strip()+'\n')
        summary_count += 1
    elif line.startswith('review/text'):
        l = line.split(':')
        f_text.write(l[1].strip()+'\n')
        text_count += 1
    
f.close()
f_text.close()
f_summary.close()
f_label.close()
