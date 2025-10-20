import re
def clean_text(s):
    s = s.replace('-\n','')           # fix hyphenated breaks
    s = re.sub(r'\n+', ' ', s)       # flatten newlines
    s = re.sub(r'Page\s*\d+', '', s, flags=re.I)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()
