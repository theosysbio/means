##################################################################
# Functions needed to reformat any power terms (x^y) within the python
# symbolic equations (e.g. base**exp) into the format used by C
# (e.g. pow(base,exp))
# This checks whether expressions involve any nested parentheses
# and makes sure these are balanced to give the correct final expressions
##################################################################

import re

# forward matches (used for identifying exponents)
no_re = re.compile('-?\d+\.?\d*')
y_re = re.compile('y_\d+')
yx_re = re.compile('yx_\d+')
c_re = re.compile('c_\d+')
brack_re = re.compile('(.*?\))')

# reverse matches (used for identifying bases if no parentheses)
noR_re = re.compile('\d*\.?\d+-?')
yR_re = re.compile('\d+_y')
yxR_re = re.compile('\d+xy')
cR_re = re.compile('\d+_c')

# reverse matches (used for identifying bases if in parentheses)
br = re.compile('(.*?\()')
brL = re.compile('\(')
brR = re.compile('\)')


# find exponent (if within parentheses; called recursively if nested brackets present)
def parse_nested_exp(e,count,exp):
    exp += brack_re.match(e).group(1)
    e = e.split(')',1)
    count += len(brL.findall(e[0]))-1
    if count != 0:
        exp = parse_nested_exp(e[1],count, exp)    
    return exp

# find exponent
def get_exp(e):
    if no_re.match(e):
        exp = no_re.match(e).group()
    if y_re.match(e):
        exp = y_re.match(e).group()
    if yx_re.match(e):
        exp = yx_re.match(e).group()
    if c_re.match(e):
        exp = c_re.match(e).group()
    if brL.match(e):
        exp = parse_nested_exp(e,0,'')
    return exp


# find base (if not within parentheses)
def get_base(b):
    if noR_re.match(b):
        base = noR_re.match(b).group()
    if yR_re.match(b):
        base = yR_re.match(b).group()
    if yxR_re.match(b):
        base = yxR_re.match(b).group()
    if cR_re.match(b):        
        base = cR_re.match(b).group()
    return base

# find base (if within parentheses; called recursively if nested brackets present)
def parse_nested_base(y, count, base):
    base += br.match(y).group(1)    
    y = y.split('(',1)
    count += len(brR.findall(y[0]))-1   
    if count != 0:
        base = parse_nested_base(y[1], count, base)           
    return base
        
# takes a string and replaces python base**exp format with C pow(base,exp) format
def replace_powers(s):
    while '**' in s:
        if ')**' in s:
            split = s.split(')**',1)
            split[0] = split[0][::-1]
            base = parse_nested_base(split[0], 1, '')
            exp = get_exp(split[1])
            split[0] = split[0].replace(base,','+base+'wop',1)
            split[1] = split[1].replace(exp, exp+')',1)
            s = split[0][::-1]+split[1]
        else:
            split = s.split('**',1)
            split[0] = split[0][::-1]
            base = get_base(split[0])
            exp = get_exp(split[1])
            split[0] = split[0].replace(base,','+base+'(wop',1)
            split[1] = split[1].replace(exp, exp+')',1)
            s = split[0][::-1]+split[1]
    return s


