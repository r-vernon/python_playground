#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:56:22 2023

@author: richard

there's a site I use that dumps a whole load of stories/articles with tags and
such but no real search functionality, just categorises by first letter of 
author, or story/article title etc... going to grab all the data so I can
build own search functionality in (e.g.) SQL

"""

# import necessary stuff
import os
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

# change directory to location of script
os.chdir(os.path.dirname(__file__))

#%%----------------------------------------------------------------------------
# read in the data
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
baseURL = baseURL.strip('\n\r') # strip newline

# set headers (copied from http://myhttpheader.com/)
headers = {'Accept-Language' : 'en-GB,en-US;q=0.9,en;q=0.8',
           'User-Agent':'Mozilla/5.0 (X11; Linux x86_64)'
                       +' AppleWebKit/537.36 (KHTML, like Gecko)'
                       +' Chrome/109.0.0.0 Safari/537.36'}

# create stringlist (A-Z)
strList = map(chr, range(ord('A'), ord('B')+1))

# create the empty lists to store data (faster than updating dataframe)
allTitle = allAuth = allDesc = allTags = allCats = allURLs = []

#%%----------------------------------------------------------------------------
# loop over every author initial (A-Z) grabbing data

for inc, currStr in enumerate(strList):
    
    # set url to access
    url = f'http://www.{baseURL}/indexes/authorindex_{currStr}.htm'
    
    # access the page and parse it
    response = requests.get(url, headers=headers)
    html_text = response.text
    soup = BeautifulSoup(html_text,'lxml')
    
    # get all stories
    links = soup.find_all('a', href=True)
    
    # get new details (URL, Desc+Tags, Title+Cat+Author)
    newDets = [(x['href'], x['title'], x.text) for x in links]
    
    # get URLs
    newURLs = [x[0] for x in newDets]
    
    # get description, tags
    # format: 'description [tag1] [...] [tagN]'
    # splitting based on '[ <any num. chars. (greedy)>]EOL'
    tmp = [re.split('(\[.*\]$)',x[1]) for x in newDets] # extract
    newDesc, newTags, *_ = [list(x) for x in zip(*tmp)] # unpack
    
    # get titles, category, author 
    # format: 'title [cat.] auth.' so splitting based on '[' or ']'
    tmp = [re.split('[\[\]]',x[2]) for x in newDets] # extract
    newTitle, newCats, newAuth = [list(x) for x in zip(*tmp)]  # unpack

    # append to list
    allTitle.append(allTitle)
    allAuth.append(newAuth)
    allDesc.append(allDesc)
    allTags.append(allTags)
    allCats.append(allCats)   
    allURLs.append(allURLs)
    
    # print just so we know progress
    print(f'Processed {currStr}')

#%%----------------------------------------------------------------------------
# format the lists nicely

# make translation table to deal with unnecesary punctuation
#   [{<+ being replaced by (((& for consistency
#   removing quotes (",'), # and *
tbl = str.maketrans('[{<]}>+','((()))&','"\'#*')

# define a function to regularise strings in a list
def regStr(strList):
    # remove or alter unnecessary punctuation
    # strip unnecessary whitespace from L&R
    # concert to lower case (casefold more aggressive)
    # remove unecessary whitespace (without regex for readability)
    strList = [x.translate(tbl).strip().casefold() for x in strList]
    strList = [' '.join(x.split()) for x in strList]
    return strList

# regularise author, description, tags, title
allAuth = regStr(allAuth)
allDesc = regStr(allDesc)
allTags = regStr(allTags)
allTitle = regStr(allTitle)

# changes tags from [tag1] [tag2] [...] [tagN] to tag1,tag2,...,tagN
#   re.findall('\[.*?\]',x)  - finds all items between square brackets
#   ','.join(...)            - joins those items as a comma sep. list
#   re.sub('[\[\]]','',...)  - strips the square brackets out
allTags = [re.sub('[\[\]]','',','.join(re.findall('\[.*?\]',x)))
          for x in allTags]

# for cat, strip whitespace, replace 'none' with 'A' (any), 
# should be one upper char.
allCats = [x.strip() for x in allCats]
allCats = ['A' if x.upper() == 'NONE' else x[0].upper() for x in allCats]

# create a  dataframe from info
df = pd.DataFrame(zip(allTitle, allAuth, allDesc, allTags, allCats, allURLs), 
                      columns=['Title', 'Author', 'Desc', 'Tags', 'Cat', 'URL'])

# save out the result
# df.to_csv('/home/richard/Documents/Python/ec_db/ec_db.csv',sep='\t',index=False)
