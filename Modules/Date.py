from datetime import datetime
import re

datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
str_date = "1st Mar 2021"
a = datetime.strptime(re.sub(r"(st|th|rd|nd)","",str_date), "%d %b %Y").strftime("%Y-%m-%d")
print(a)

# get datetime.datetime obj, obj.strftime([format]) return str
c = datetime.now()
print(datetime.now())
print(type(c))
y = c.strftime("%Y-%m-%d")
print("Y", type(y))

# get string, datetime.strptime([string], [format code]) return datetime.datetime
str_date = "1 Mar 2021"
b = datetime.strptime(str_date, "%d %b %Y")
print(type(b))