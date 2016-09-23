#! /usr/bin/env python3


import os
import sys
import re



data = [[1,3],[2,8],[3,15]]
#true_y = [3,8,15]

lr = 0.001

a = 0
b = 0

eta = 100

for i in range(10000):
    now_itm = i % 3
    x = data[now_itm][0]
    y = data[now_itm][1]
    t_y = a * x * x + b * x
    dy = -2 * (y - t_y)
    a -= lr * (dy * (x * x) + eta * (4 * a - 4 * b))
    b -= lr * (dy * (x) + eta * (4 * b - 4 * a))
    xent = 0
    reg = 0
    for j in range(len(data)):
        x = data[j][0]
        t_y = a * x * x + b * x
        xent += (t_y - data[j][1]) * (t_y - data[j][1])
        reg += (a - b) * (a - b) * 2
    if True:
        print(xent, reg, xent + eta * reg, a, b)
