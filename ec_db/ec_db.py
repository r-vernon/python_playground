#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:56:22 2023

@author: richard
"""

import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

# change directory to location of script
os.chdir(os.path.dirname(__file__))

#%% read in the data
'''
Format of each story:
<a href="http://www.TargetSite.org/view.php?storyid=1234" target="_new" 
    title="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do 
    eiusmod tempor incididunt ut labore et dolore magna aliqua [tag1] [tag2] 
    [...] [tagN]">This Is The Story Title [Cat.]
    <span class="indexauthor">Author Name</span></a>
    
Want to grab (in order of appearance):
    URL: http://www.TargetSite.org/view.php?storyid=1234
    Description: Lorem ipsum dolor sit amet, consectetur adipiscing elit, ...
    Tags: [tag1] [tag2] [...] [tagN]
    Title: This Is The Story Title
    Category: [Cat.]
    Author: Author Name
'''

# get the base url
# (reading it in rather than specifying it as site may be legally 'grey' XD)
with open('baseURL','r') as f:
    baseURL = f.readline()

# set headers (copied from http://myhttpheader.com/)
headers = {'Accept-Language' : 'en-GB,en-US;q=0.9,en;q=0.8',
           'User-Agent':'Mozilla/5.0 (X11; Linux x86_64)'
                       +' AppleWebKit/537.36 (KHTML, like Gecko)'
                       +' Chrome/109.0.0.0 Safari/537.36'}

# create stringlist (A-Z)
strList = list(map(chr, range(ord('A'), ord('Z')+1)))

# create empty dataframe
cols=['Title', 'Author', 'Desc', 'Tags', 'Cat', 'URL']
df = pd.DataFrame(columns=cols)

inc = 0
currStr = 'A'

# for inc, currStr in enumerate(strList):
    
# set url to access
url = f'http://www.{baseURL}/indexes/authorindex_{currStr}.htm'

# access the page and parse it
response = requests.get(url, headers=headers)
html_text = response.text
soup = BeautifulSoup(html_text,'lxml')

# get all stories
links = soup.find_all('a', href=True)

# get all details (URL, Desc+Tags, Title+Cat+Author)
allDets = [(x['href'], x['title'], x.text) for x in links]

# get URLs
allURL = [x[0] for x in allDets]

# get description, tags
tmp = [re.split('(\[.*\]$)',x[1]) for x in allDets] # extract
allDesc, allTags, *_ = [list(x) for x in zip(*tmp)] # unpack
del tmp

# strip any whitespace from descriptions
allDesc = [x.strip() for x in allDesc]

# changes tags from [tag1] [tag2] [...] [tagN] to tag1,tag2,...,tagN
#   re.findall('\[.*?\]',x)  - finds all items between square brackets
#   ','.join(...)            - joins those items as a comma sep. list
#   re.sub('[\[\]]','',...)  - strips the square brackets out
allTags = [re.sub('[\[\]]','',','.join(re.findall('\[.*?\]',x)))
          for x in allTags]

# get titles, category, author (splitting based on '[' or ']')
tmp = [re.split('[\[\]]',x[2]) for x in allDets] # extract
allTitle, allCat, allAuth = [list(x) for x in zip(*tmp)]  # unpack
del tmp

# strip any whitespace from title, author
allTitle = [x.strip() for x in allTitle]
allAuth = [x.strip() for x in allAuth]

# create a new dataframe from info
new_df = pd.DataFrame(zip(allTitle, allAuth, allDesc, 
                          allTags, allCat, allURL), 
                      columns=cols)

# concate with old dataframe
df = pd.concat([df,new_df],ignore_index=True)

# print just so we know progress
print(f'Processed {currStr}')

# save out the result
# df.to_csv('/home/richard/Documents/Python/ec_db/ec_db.csv',sep='\t',index=False)
