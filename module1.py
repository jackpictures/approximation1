import numpy as np
import math


def print_poly(a1):
    polll = list(map(float, a1))
    st = ''
    for i in range(len(polll)):
        if (float(polll[i])) != 0.0:
            if ((i)) > 1:
                if (float(polll[i]) > 0):
                    st += '+' + str(polll[i])
                    st += '*x^'
                    st += str((i))
                else:
                    st += str(polll[i])
                    st += '*x^'
                    st += str((i))
            if ((i)) < 1:
                if (float(polll[i]) > 0):
                    st += '+' + str(polll[i])
                else:
                    st += str(polll[i])
            if ((i)) == 1:
                if (float(polll[i]) > 0):
                    st += '+' + str(polll[i])
                    st += '*x'
                else:
                    st += str(polll[i])
                    st += '*x'
    for i in range(len(st)):
        if (i < len(st) - 3):
            if (st[i] == '1') and (st[i + 3] == "*" and (st[i+2]=='0')) and (st[i-1]=='+' or st[i-1]=='-'):
                st = st[:i] + st[(i + 4):]
    if len(st)==0:
        return 0.0
    if (st[0] == '+'):
        return st[1:]
    else:
        return st

def upper(a,nums):
    upper = 0
    l=[]
    for i in range(len(nums)):
        k = float(nums[i])
        if k == 0:continue
        k*=-1
        k_poly = np.array(list([1,k]),dtype = float)
        div = np.polydiv(a, k_poly)
        flag = 1
        for j in range(len(div[0])):
            if (float(div[0][j])<0): flag = 0
        if (float(div[1])<0): flag = 0
        if flag:
            upper = k
            break
    return upper

def sign(num):
    return -1 if num < 0 else 1

def half_divide_method(a, b, f,a1,e = 0.001):
    x = (a + b) / 2
    i=0
    flag = 0
    while math.fabs(f(x,a1)) >= e:
        if i>1000: 
            flag = 1
            break
        x = (a + b) / 2
        i+=1
        a, b = (a, x) if f(a,a1) * f(x,a1) < 0 else (x, b)
    if flag==0: return (a + b) / 2
