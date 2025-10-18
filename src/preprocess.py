import re
def clean_text(s):
    s = s.replace('-\n','')           # hyphen line breaks
    s = s.replace('\n',' ')           # flatten lines
    s = re.sub(r'\bPage\s*\d+\b','',s, flags=re.I)
    s = re.sub(r'\s{2,}',' ',s)
    return s.strip()
