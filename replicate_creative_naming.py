#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:18:29 2022

replicate creative naming task...?

@author: sskalicky
"""

import pandas as pd

# install custom spacy wrapper for google USE
# https://github.com/MartinoMensio/spacy-universal-sentence-encoder
# https://stackoverflow.com/questions/52113939/spacy-strange-similarity-between-two-sentences

# the package
# !pip install git+https://github.com/MartinoMensio/spacy-universal-sentence-encoder.git

# the actual langauge model
# !pip install https://github.com/MartinoMensio/spacy-universal-sentence-encoder/releases/download/v0.4.3/xx_use_lg-0.4.3.tar.gz#xx_use_lg-0.4.3

# it would be good to fork the repo as it is - for replicability etc. 

import spacy_universal_sentence_encoder
nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')

# load in creative naming data
dat = pd.read_csv('creative_naming_raw_answers.csv')

# filter out the off task participants
dat = dat[dat['on.task'] == 1]

# check columns
dat.columns

# create language subgroups
nes_dat = dat[dat['group'] == 'NES']
rom_dat = dat[dat['group'] == 'ROM']

# slice out their answer/items pairs
# i've already manually cleaned/pre-processed the answers externally to python
nes_answers = nes_dat[['cleaned_answer', 'item']]
rom_answers = rom_dat[['cleaned_answer', 'item']]

# create list of items
items = list(dat['item'].unique())
items

# write function to calculate similarity
from collections import defaultdict

def sim_dict(answers, items):
  
  # the eventual output  
  output_dict = defaultdict(dict)
  
  
  for item in items:
    
     # grab every answer made for each item
    item_answers = list(answers[answers['item'] == item]['cleaned_answer'])
    
    print(f'computing similarity for answers made for {item}')
    
    # for each answer, compute sentence similarity to all other answers for same item
    for index, answer in enumerate(item_answers):
      print(f'computing similarity for {answer}...')
      print(f'{index+1} of {len(item_answers)}')
      answer_sims = [(x, nlp(answer).similarity(nlp(x))) for x in item_answers]
      output_dict[item][answer] = dict(answer_sims)

  return output_dict

# Similarities for mTurk participants
nes_sims = sim_dict(nes_answers, items)

# check out some answers - output is a dictionary with {answer: other answers}
nes_sims.keys()
nes_sims['wallet'].keys()
nes_sims['wallet']['butt file folder']


# save sims to a pickle
import pickle
afile = open('nes_sims.pkl', 'wb')
pickle.dump(nes_sims, afile)
afile.close()

# get similarities for Romanian participants
rom_sims = sim_dict(rom_answers,items)

# save sims to a pickle
bfile = open('rom_sims.pkl', 'wb')
pickle.dump(rom_sims, bfile)
bfile.close()


# Now return this data into a .csv for analysis

import statistics

nes_averages = defaultdict(dict)

# first calculate averages for each item/answer
# the average is the average sim for that answer to all other answers 
for item in nes_sims.keys():
  for answer in nes_sims[item].keys():
    values = [v for v in nes_sims[item][answer].values()]
    nes_averages[item][answer] = statistics.mean(values)
    
# check some
nes_averages['balloon']['air ball']
nes_averages['wallet']['butt file folder']

# do the same for Romanians
rom_averages = defaultdict(dict)

for item in rom_sims.keys():
  for answer in rom_sims[item].keys():
    values = [v for v in rom_sims[item][answer].values()]
    #rom_averages.append(['English L2', item, answer, statistics.mean(values)])
    rom_averages[item][answer] = statistics.mean(values)

rom_averages


# now we want to join this with the original data (dat)
# the reason we need to do this is because some participants repeated answers, and thus the dictionary will not map neatly to the original csv

# I swear I hate pandas warnings so much
raw_dat_nes = dat.loc[(dat['group'] == 'NES') & (dat['on.task'] == 1)].copy()
raw_dat_rom = dat.loc[(dat['group'] == 'ROM') & (dat['on.task'] == 1)].copy()


# add averages to nes data
for index, row in raw_dat_nes.iterrows():
  item = raw_dat_nes.loc[index, 'item']
  answer = raw_dat_nes.loc[index, 'cleaned_answer']

  raw_dat_nes.loc[index, 'avg_sim'] = nes_averages[item][answer]
  
  
raw_dat_nes
  
# add averages to rom data
for index, row in raw_dat_rom.iterrows():
  item = raw_dat_rom.loc[index, 'item']
  answer = raw_dat_rom.loc[index, 'cleaned_answer']

  raw_dat_rom.loc[index, 'avg_sim'] = rom_averages[item][answer]
  

avg_sims = pd.concat([raw_dat_nes, raw_dat_rom])
avg_sims.columns

# save the output csv...
avg_sims.to_csv('average_similarities.csv')
