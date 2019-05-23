#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:07:00 2019

@author: emre
"""
import pandas as pd

chicago2018 = pd.read_csv('./chicago2018.csv')
df = pd.DataFrame(columns=['ages'])

#for i,age in enumerate(london2018['ages']):
#    if age == '85+' or age == '80-84' or age == '75-79' or age == '70-74':
#        london2018['ages'][i] =  '70+'

chicago2018 = chicago2018.replace('16-19', '18-39')
chicago2018 = chicago2018.replace('20-24', '18-39')
chicago2018 = chicago2018.replace('25-29', '18-39')
chicago2018 = chicago2018.replace('30-34', '18-39')
chicago2018 = chicago2018.replace('35-39', '18-39')

chicago2018 = chicago2018.replace('70-74', '70+')
chicago2018 = chicago2018.replace('75-79', '70+')
chicago2018 = chicago2018.replace('80+', '70+')

#for i,age in enumerate(chicago2018['ages']):
#    if age == '16-19' or age == '20-24' or age == '25-29' or age == '30-34' or age == '35-39':
#        chicago2018['ages'][i] =  '18-39'
chicago2018.to_csv('./chicago2018_model.csv')