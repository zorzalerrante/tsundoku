import re
import string

from tsundoku.utils.files import read_list

URL_RE = re.compile(
    r"(https?://[-A-Za-z0-9+&@#/%?=~_()|!:,.;]*[-A-Za-z0-9+&@#/%=~_()|])"
)
PUNCTUATION_RE = re.compile("^[{}…¿¡“”]+$".format(string.punctuation))


def build_re_from_files(
    filename, sort=False, re_flags=re.IGNORECASE, filter_func=None, process_func=None
):
    if type(filename) == str:
        terms = read_list(filename)

    elif type(filename) in (list, map):
        terms = []
        for fn in filename:
            terms.extend(read_list(fn))

    else:
        raise ValueError("Unknown type for terms: {}".format(type(filename)))

    if filter_func is not None:
        terms = [t for t in terms if filter_func(t)]

    if process_func is not None:
        terms = list(map(process_func, terms))

    if sort:
        terms = sorted(set(terms))

    return re.compile("|".join(terms), re_flags)
