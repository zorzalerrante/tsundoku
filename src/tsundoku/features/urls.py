from urllib.parse import urlparse
import re

DOMAIN_PREFIXES = re.compile(
    "^(?:w+|web|open|m|s|i|es|amp|\w{2}-\w{2}|ar|cl|profile|my|mobile)\."
)

DISCARD_URLS = [
    "",
    "twitter.com",
    "bit.ly",
    "ow.ly",
    "buff.ly",
    "fb.me",
    "goo.gl",
    "g.co",
]


def get_domain(url):
    link = urlparse(url).netloc
    return DOMAIN_PREFIXES.sub("", link).lower()
