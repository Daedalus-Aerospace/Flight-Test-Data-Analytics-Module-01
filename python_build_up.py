# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 07:07:31 2023

@author: user
"""

# Python data types and assignment

# float
# assign a name to the float value
# acceleration due to gravity
g = 9.81

# Boolean keywords: False, True
# Weight on Wheels
wow = True

# int
# Row index number
idx = 3054

# string
# Filename
f_name = "TAS_Conv.xlsx"

# python data structures

# list
# explain index
# open in the Spyder Variable Explorer
# explain list index starts from 0
Nz = ["1g", "2g", "3g", "4g", "5g", "6g"]
# access list
Nz[4]
# splice list
# explain start index is inclusive, end index is exclusive
Nz[:3]
Nz[3:]
Nz[2:5]


# explain changing variable values
TAS_kt_list = [100, 200, 300, 400, 500, 600]

# convert all elements of a list
# Kt to mps

# bias error in data: add by a factor
TAS_kt_list + 10
# access individual element of list
TAS_kt_list[0] + 10

# gain error in data: multiply by factor
TAS_kt_list * 1.1

# access single element of a list
TAS_kt_list[0]
print(TAS_kt_list[0])
print(TAS_kt_list[1])
print(TAS_kt_list[2])

TAS_kt_list[0] + 10
TAS_kt_list[0] * 2
TAS_kt_list[0] * 1.2

# access each and every individual element of a list
# and then apply calculations

TAS_mps = []
for item in TAS_kt_list:
    TAS_mps.append(item * 0.5144)


# "list comprehension" a short form for simple loops
TAS_mps = [x * 0.5144 for x in TAS_kt_list]


# define a function
def kt2mps(x):
    """Convert airspeed from knots to meters per second."""
    return x * 0.5144


kt2mps(TAS_kt_list)
TAS_mps = [kt2mps(x) for x in TAS_kt_list]

import pandas as pd

df = pd.DataFrame(TAS_kt_list, columns="TAS(kt)")
df["TAS_mps"] = df[["TAS(kt)"]].apply(kt2mps)


df.to_excel("TAS_Conv.xlsx")
df.to_excel("TAS_Conv.xlsx", index=False)

df1 = pd.read_excel("TAS_Conv.xlsx")


# Tuple
TAS_kt_tup = (100, 200, 300, 400, 500, 600)
# convert a list to tuple
tuple(df1["TAS(kt)"])
list(tuple(df1["TAS(kt)"]))

# different types of brackets
# Square brackets: []
# Parentheses: () (AKA "round brackets")
# Curly braces: {}

df_dict = df1.to_dict("dict")
df_dict["TAS(kt)"].values()
list(df_dict["TAS(kt)"].values())

# Angle brackets: <>
