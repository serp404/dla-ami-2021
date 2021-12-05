import re

_whitespace_re = re.compile(r'\s+')
_abbreviations_re = [
    (re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1])
    for x in [
        ('mrs', 'misess'),
        ('mr', 'mister'),
        ('dr', 'doctor'),
        ('st', 'saint'),
        ('co', 'company'),
        ('jr', 'junior'),
        ('maj', 'major'),
        ('gen', 'general'),
        ('drs', 'doctors'),
        ('rev', 'reverend'),
        ('lt', 'lieutenant'),
        ('hon', 'honorable'),
        ('sgt', 'sergeant'),
        ('capt', 'captain'),
        ('esq', 'esquire'),
        ('ltd', 'limited'),
        ('col', 'colonel'),
        ('ft', 'fort')
    ]
]


def process_abbreviations(text):
    for regex, replacement in _abbreviations_re:
        text = re.sub(regex, replacement, text)
    return text


def process_uppercase(text):
    return text.lower()


def process_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def basic_cleanup(text):
    text = process_uppercase(text)
    text = process_abbreviations(text)
    text = process_whitespace(text)
    return text
