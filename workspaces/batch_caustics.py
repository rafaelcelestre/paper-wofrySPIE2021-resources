#!/usr/bin/python
# coding: utf-8

import os

E = 8.2          # Eu-XFEL energy
cst_pts = 4000
n = 5
cst = 100e-3
d = 0

stack = 1   # stack 0 (ideal lenses); stack 1 (R=50um); stack 2(R=100um)
cmd = 'python srw_config_01.py -s False -p True -c False -e %f -stck %d -n %d -cr %f -cp %d -prfx "5x50um_Be_CRL" ' \
      '-dir "results" -d %f' % (E, stack, n, cst, cst_pts, d)
print(cmd)
os.system(cmd)
