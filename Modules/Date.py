from datetime import datetime
import re

# datetime.strptime: string to datetime.datetime
_format = '%b %d %Y %I:%M%p'
datetime_object = datetime.strptime('Jun 1 2005  1:33PM', _format)

# datetime.strftime: datetime.datetime to string
# str_date -> re.sub r"(st|th|rd|nd)" -> %d %b %Y -> %Y-%m-%d
str_date = "1st Mar 2021"
str_date_cut = re.sub(r"(st|th|rd|nd)", "", str_date)
datetime_obj = datetime.strptime(str_date_cut, "%d %b %Y")
# cut the HH MM SS
str_date_without_time = datetime_obj.strftime("%Y-%m-%d")

diff = datetime.now() - datetime.strptime("10-11-2021:11:30", "%m-%d-%Y:%H:%M")
print("time now", datetime.now())
diff_from_now = datetime.now() - datetime_obj
print("time delta days", diff.days)
print("time delta seconds", diff.seconds)
print("time delta {0} min and {1} seconds:".format(*divmod(diff.seconds, 60)))

# get datetime.datetime obj, obj.strftime([format]) return str
c = datetime.now()
y = c.strftime("%Y-%m-%d")

# get string, datetime.strptime([string], [format code]) return datetime.datetime
str_date = "1 Mar 2021"
b = datetime.strptime(str_date, "%d %b %Y")

t1 = "06:30"
t2 = "23:20"
t1_datetime = datetime.strptime(t1, "%H:%M")
t2_datetime = datetime.strptime(t2, "%H:%M")
diff = t2_datetime - t1_datetime
hour, rem = divmod(diff.seconds, 3600)
print("hour, rem:", hour, rem)
min, rem = divmod(rem, 60)
print("min, rem:", min, rem)

