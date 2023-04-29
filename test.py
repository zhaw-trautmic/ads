from pygooglenews import GoogleNews

gn = GoogleNews()
s = gn.search('boeing OR airbus')

for entry in s["entries"]:
    print(entry["title"])
view rawpygooglenews.py hosted with ❤ by GitHub