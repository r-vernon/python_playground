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
from unidecode import unidecode

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
strList = map(chr, range(ord('A'), ord('Z')+1))

# create the empty lists to store data (faster than updating dataframe)
allTitle, allAuth, allDesc, allTags, allCats, allURLs = ([] for x in range(6))

#%%----------------------------------------------------------------------------
# loop over every author initial (A-Z) grabbing data

for currStr in strList:
    
    # set url to access
    url = f'http://www.{baseURL}/indexes/authorindex_{currStr}.htm'
    
    # access the page and parse it
    html_text = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html_text,'lxml')
    
    # get all stories
    links = soup.find_all('a', href=True)
                    
    # get new details format: (URL, (Desc+Tags, Title+Cat+Author))
    newURLs,newDets = zip(*[(x['href'],(x['title'], x.text)) for x in links])
    
    # add new URLs to current list
    allURLs.extend(newURLs)
    
    # loop over newDets (more readable than list comprehension)
    for currDet in newDets:

        # get description, tags
        # format: 'description [tag1] [...] [tagN]'
        # splitting based on '[any num. chars. (greedy)]EOL'
        tmp = re.split('(\[.*\]$)',currDet[0])
        allDesc.append(tmp[0])
        allTags.append(tmp[1])
        
        # get titles, category, author 
        # format: 'title [cat.] auth.' so splitting based on '[' and ']'
        tmp = re.split('[\[\]]',currDet[1])
        allTitle.append(tmp[0])
        allCats.append(tmp[1])
        allAuth.append(tmp[2])    
    
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
    # remove or alter unnecessary punctuation (translate)
    # strip unnecessary outer whitespace from L&R
    # convert to lower case (casefold more aggressive)
    # remove unecessary inner whitespace (without regex for readability)
    # use unidecode to ensure characters are ASCII friendly
    strList = [x.translate(tbl).strip().casefold() for x in strList]
    strList = [' '.join(x.split()) for x in strList]
    strList = [unidecode(x) for x in strList]
    return strList

# regularise title author, description, tags
allTitle = regStr(allTitle)
allAuth = regStr(allAuth)
allDesc = regStr(allDesc)
allTags = regStr(allTags)

# changes tags from (tag1) (tag2) (...) (tagN) to tag1,tag2,...,tagN
#   (regularised hence (tag1) not [tag1])
#   re.findall('\(.*?\)',x)  - finds all items between brackets
#   sorted(...)              - sorts the items
#   ','.join(...)            - rejoins those items as a comma seperated list
#   re.sub('[\(\)]','',...)  - strips the brackets 
allTags = [re.sub('[\(\)]','',','.join(sorted(re.findall('\(.*?\)',x))))
          for x in allTags]

# for categories
#   strip whitespace, convert to upper case
#   replace 'none' with 'A' (any), make sure the rest are one char
allCats = [x.strip().upper() for x in allCats]
allCats = ['A' if x == 'NONE' else x[0] for x in allCats]

# create a  dataframe from info
df = pd.DataFrame(zip(allTitle, allAuth, allDesc, allTags, allCats, allURLs), 
                      columns=['Title', 'Author', 'Desc', 'Tags', 'Cat', 'URL'])

# save out the result
df.to_csv('ec_db.csv',sep='\t',index=False)
