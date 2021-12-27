import datetime
import re

import pytz

twitter_date_format = "%a %b %d %H:%M:%S +0000 %Y"
old_twitter_date_format = "%Y-%m-%d %H:%M:%S"


def parse_twitter_date(text, dst_timezone):
    try:
        naive_datetime = datetime.datetime.strptime(
            text, twitter_date_format
        )  # .replace(tzinfo=pytz.UTC)
    except ValueError:
        naive_datetime = datetime.datetime.strptime(
            text, old_twitter_date_format
        )  # .replace(tzinfo=pytz.UTC)
    dt = pytz.utc.localize(naive_datetime).astimezone(dst_timezone)
    return dt  # .strftime(self.target_date_format)


def date_from_filename(fname, index=-1):
    return re.findall("(\d{4}-\d{2}-\d{2})", fname)[index]
