from pygooglenews import GoogleNews
import json
import time

gn = GoogleNews()
s = gn.search('boeing OR airbus')

for entry in s["entries"]:
    print(entry["title"])
