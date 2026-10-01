"""Versioned finite equivalence library for integer answers 0..99.

This is a declared set of surface forms, not a universal semantic detector.
No modular equivalence, arbitrary expressions, Roman numerals, or homophones:
those require context that a vocabulary item does not provide.
"""
from decimal import Decimal
VERSION = 'integer-surfaces-v1'
UNITS = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
TENS = ('zero ten twenty thirty forty fifty sixty seventy eighty ninety').split()

def words(n):
    if type(n) is not int or not 0<=n<=99:raise ValueError('Expected integer0..99')
    return UNITS[n] if n<20 else TENS[n//10]+('-'+UNITS[n%10] if n%10 else '')

def forms(n):
    word=words(n)
    english={word,word.replace('-',' ')}
    english={v for w in english for v in (w,w.capitalize(),w.upper())}
    numeric={str(n),'+'+str(n),str(n)+'.0',str(n)+'.00',str(n)+'/1'}
    # Fullwidth decimal digits are unambiguous numeric characters.
    numeric.add(str(n).translate(str.maketrans('0123456789','０１２３４５６７８９')))
    # Space belongs to the spelling variant, not a standalone equivalent token.
    return sorted({prefix+s for s in english|numeric for prefix in ('',' ')})

def tokenize(n,tok):
    sequences={tuple(tok.encode(s,add_special_tokens=False)) for s in forms(n)}
    if not sequences or any(not s for s in sequences):raise ValueError('Empty alias')
    # A short alias event already covers a longer alias with precisely that token
    # prefix. Remove nested events to avoid double-counting their probability.
    ordered=sorted(sequences,key=lambda s:(len(s),s));result=[]
    for seq in ordered:
        if not any(seq[:len(old)]==old for old in result):result.append(seq)
    return [list(s) for s in result]

def manifest():
    return dict(version=VERSION,domain=[0,99],forms={str(n):forms(n) for n in range(100)},
                limitations='Finite surface library; arbitrary encodings not covered. Lens probabilities are sequence-prefix readouts, not guaranteed exact semantic events. Multi-token aliases are scored jointly with forced alias prefixes, never by summing fragment probabilities.')
