"""Answer-independent numeric-token mask, including conservative subword pieces."""
import re,unicodedata
from .equivalents import UNITS,TENS
VERSION='all-numeric-tokens-v1'
NUMBER_WORDS=set(UNITS+TENS+('hundred thousand million billion trillion quadrillion quintillion sextillion septillion octillion nonillion decillion zeroes zeros oh nil nought naught infinity infinite nan pi tau dozen half halves quarter quarters third thirds fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth hundredth thousandth millionth billionth first second zeroth double triple quadruple positive negative plus minus percent percentage fraction decimal numerator denominator').split())
SYMBOLS=set('+-−±∓×÷/=≠≈<>≤≥%‰‱∞√∛∜∑∏^*.,:⁄∕⋅·∝∫∂∆ΔπΠτΤℕℤℚℝℂ')
ROMAN=re.compile(r'M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})',re.I)

def reasons(text):
    s=text.strip();result=[]
    if any(c.isnumeric() for c in s):result.append('Unicode numeric character')
    normalized=unicodedata.normalize('NFKC',s)
    if any(c.isnumeric() for c in normalized) and not result:result.append('normalized numeric character')
    if any(w.casefold() in NUMBER_WORDS for w in re.findall(r'[^\W\d_]+',normalized)):
        result.append('number word')
    if s and all(c in SYMBOLS or c.isspace() for c in s):result.append('numeric/math punctuation')
    if s and ROMAN.fullmatch(s):result.append('Roman numeral; context-ambiguous forms included')
    return result

def build(tok):
    selected={}
    for i in range(len(tok)):
        if i in tok.all_special_ids:continue
        text=tok.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        rs=reasons(text)
        if rs:selected[i]=dict(id=i,text=text,reasons=rs)
    # Include component tokens when a number word requires multiple BPE tokens.
    # This intentionally also suppresses those fragments in nonnumeric contexts.
    for word in sorted(NUMBER_WORDS):
        for form in (word,word.capitalize(),word.upper()):
            for prefix in ('',' '):
                for i in tok.encode(prefix+form,add_special_tokens=False):
                    if i in tok.all_special_ids or not tok.decode([i]).strip():continue
                    row=selected.setdefault(i,dict(id=i,text=tok.decode([i],clean_up_tokenization_spaces=False),reasons=[]))
                    if 'number-word subtoken' not in row['reasons']:row['reasons'].append('number-word subtoken')
    for digit in '0123456789':
        assert all(i in selected for i in tok.encode(digit,add_special_tokens=False))
    return dict(version=VERSION,token_ids=sorted(selected),tokens=[selected[i] for i in sorted(selected)],
        number_words=sorted(NUMBER_WORDS),math_symbols=''.join(sorted(SYMBOLS)),
        scope='Entire tokenizer vocabulary; same mask for every problem and architecture. Unicode numbers, number words/subtokens, standalone numeric/math punctuation, Roman numerals.',
        limitations='Conservative lexical definition: fragments, punctuation and Roman letters also have nonnumeric uses. Arbitrary learned codes or all possible linguistic numeric expressions are not enumerable.')
