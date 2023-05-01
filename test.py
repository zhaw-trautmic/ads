from pygooglenews import GoogleNews

gn = GoogleNews()
s = gn.search('boeing OR airbus')

for entry in s["entries"]:
    print(entry["title"])
