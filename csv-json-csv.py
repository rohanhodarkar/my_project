# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:38:01 2018

@author: rohanh
"""

import csv
import json


with open('tf_test_n.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)


with open('tf_test.json', 'w') as f:
    for row in rows:
        json.dump(row, f)
        f.write('\n')
        

ifile = open('merchant_data.csv')

reader=csv.reader(ifile)    

reader2=csv.DictReader(ifile)

rows=list(reader2)


for row in rows:
    print(row)

#############JSON-TO-CSV###############

import json, csv
infile = open("merchant_data_10.json", "r")
outfile = open("merchant_churn_361.csv", "w")
writer = csv.writer(outfile)
for r in infile.readlines():
    writer.writerow(json.load(r.read()))
    #outfile.write('\n')
    print(r)
writer=csv.writer(outfile)
writer.

import json, csv
infile = open("foo.json", "r")
outfile = open("bar.csv", "w")
writer = csv.writer(outfile)
for row in json.loads(infile.read()):
    writer.write(row)


