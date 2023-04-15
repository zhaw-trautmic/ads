#import modules
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

#get the URL using response variable
my_url = "https://finance.yahoo.com/news"
response = requests.get(my_url)

#Catching Exceptions
print("response.ok : {} , response.status_code : {}".format(response.ok , response.status_code))
print("Preview of response.text : ", response.text[:500])

# Get current working directory
print(os.getcwd())

#utility function to download a webpage and return a beautiful soup doc
def get_page(url):
    response = requests.get(url)
    if not response.ok:
        print('Status code:', response.status_code)
        raise Exception('Failed to load page {}'.format(url))
    page_content = response.text
    doc = BeautifulSoup(page_content, 'html.parser')
    return doc

#function call
doc = get_page(my_url)

#appropritae tags common to news-headlines to filter out the necessary information.
a_tags = doc.find_all('a', {'class': "js-content-viewer"})
print(len(a_tags))

#print(a_tags[1])
news_list = []

#print top 10 Headlines
for i in range(1,len(a_tags)+1):
    news = a_tags[i-1].text
    news_list.append(news)
    print("Headline "+str(i)+ ":" + news)
    news_df = pd.DataFrame(news_list)
    news_df.to_csv('Market_News')
    
    import os
import platform
import socket
from platform import python_version
from datetime import datetime

print('-----------------------------------')
print(os.name.upper())
print(platform.system(), '|', platform.release())
print('Datetime:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Python Version:', python_version())
print('-----------------------------------')