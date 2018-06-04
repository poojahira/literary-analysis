from bs4 import BeautifulSoup
import requests
import csv
import sys
from urllib.request import urlopen

def get_content(filename):
    str1 = ''
    page = urlopen(filename)
    soup = BeautifulSoup(page, 'html.parser')
    soup.span.decompose()
    story = soup.find('div',attrs={'class':'ArticleBody__articleBody___1GSGP'})
    for para in story.find_all("p"):
        str1 = str1 + str(para)
    author = soup.find('p',attrs={'class':'Byline__by___37lv8'})
    author = author.a.get_text()
    time = soup.find('time',attrs={'class':'IssueDate__issueDate___2e_OC'})
    time = time.get_text()
    title = soup.find('h1',attrs={'class':'ArticleHeader__hed___GPB7e'})
    title = title.get_text()
    tags = soup.find_all('li',attrs={'class':'Tags__item___3xh9I'})
    if len(tags) == 1:
        tag1 = tags[0].get_text()
    if len(tags) == 2:
        tag1 = tags[0].get_text()
        tag2 = tags[1].get_text()
    elif len(tags) == 3:
        tag1 = tags[0].get_text()
        tag2 = tags[1].get_text()
        tag3 = tags[2].get_text()
    elif len(tags) == 4:
        tag1 = tags[0].get_text()
        tag2 = tags[1].get_text()
        tag3 = tags[2].get_text()
        tag4 = tags[3].get_text()
    elif len(tags) == 5:
        tag1 = tags[0].get_text()
        tag2 = tags[1].get_text()
        tag3 = tags[2].get_text()
        tag4 = tags[3].get_text()
        tag5 = tags[4].get_text()
    elif len(tags) == 6:
        tag1 = tags[0].get_text()
        tag2 = tags[1].get_text()
        tag3 = tags[2].get_text()
        tag4 = tags[3].get_text()
        tag5 = tags[4].get_text()
        tag6 = tags[5].get_text()
    link = soup.find('link',attrs={'rel':'canonical'})
    link = link['href']
    
    with open('articles.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)
        if len(tags) == 0:
            writer.writerow([title,str1,author,time,link])
        elif len(tags) == 1:
            writer.writerow([title,str1,author,time,link,tag1])
        elif len(tags) == 2:
            writer.writerow([title,str1,author,time,link,tag1,tag2])
        elif len(tags) == 3:
            writer.writerow([title,str1,author,time,link,tag1,tag2,tag3])
        elif len(tags) == 4:
            writer.writerow([title,str1,author,time,link,tag1,tag2,tag3,tag4])
        elif len(tags) == 5:
            writer.writerow([title,str1,author,time,link,tag1,tag2,tag3,tag4,tag5])
        elif len(tags) == 6:
            writer.writerow([title,str1,author,time,link,tag1,tag2,tag3,tag4,tag5,tag6])
    csv_file.close()

get_content(sys.argv[1])
	
