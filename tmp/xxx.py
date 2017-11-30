#!/usr/bin/env python
from os import listdir
for filename in listdir("data"):
    if not filename.endswith(".csv"): continue
    print filename
