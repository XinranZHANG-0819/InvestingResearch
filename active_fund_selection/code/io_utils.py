"""Loading helpers: repair double-encoded UTF-8 text in some Drive exports."""
import gzip, io, glob, os
import pandas as pd

DL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dl')

def _text(path):
    raw = open(path, 'rb').read()
    if path.endswith('.gz'):
        raw = gzip.decompress(raw)
    t = raw.decode('utf-8-sig')
    if 'ä¸' in t[:20000] or 'ï»¿' in t[:10]:
        t = t.encode('latin-1').decode('utf-8-sig')
    return t.lstrip('﻿')

def load(prefix, **kw):
    path = glob.glob(os.path.join(DL, prefix + '*', '*'))[0]
    return pd.read_csv(io.StringIO(_text(path)), dtype=str, low_memory=False, **kw)

def kv(note):
    out = {}
    if isinstance(note, str):
        for part in note.split(';'):
            if '=' in part:
                k, v = part.split('=', 1)
                out[k.strip()] = v.strip()
    return out
