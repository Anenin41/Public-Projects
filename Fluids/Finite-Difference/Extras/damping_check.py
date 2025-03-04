# Code that calculates the damping for Exercise 4b - Practical 4 #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com
# Created: Fri 15 Nov 2024 @ 17:25:01 +0100
# Modified: Tue 04 Mar 2025 @ 19:31:06 +0100

# Local Critical Points
# x array stores the omega = 0.5 & dt = 0.5 numerical setting
# y array stores the omega = 0.6 & dt = 0.5 numerical setting
x = [1, 0.0409907, 0.172296, 0.0961056, 0.0951366, 0.080782, 0.0728483, 0.0657287, 0.0601289]

y = [1, 0.137129, 0.12383875, 0.0104539, 0.0234849, 0.00856998, 0.0066252, 0.00354584, 0.00229428]

# These arrays store the ratios between successive amplitudes (arrays x & y)
r1 = []
r2 = []

for i in range(1,9):
    dummy1 = x[i] / x[i-1]
    r1.append(dummy1)
    
    dummy2 = y[i] / y[i-1]
    r2.append(dummy2)

# Average the amplitude ratios to get a rough (sample) approximation of the damping
damping1 = sum(r1)/len(r1)
damping2 = sum(r2)/len(r2)

print("Damping of the 1st Setting: ", damping1)
print("Damping of the 2nd Setting: ", damping2)
