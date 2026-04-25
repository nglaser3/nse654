import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc_file("../matplotlib.rc")

import numpy as np

x = np.linspace(0,10, 1000)
y = np.sin(x/4) * (8/3 + x**2 / 50)

plt.plot(x, y)
plt.xlabel("Slab Depth  $[cm]$")
plt.ylabel(r"Scalar Flux $[n\cdot cm^{-2}]$")
plt.savefig("hw1q1a.pdf")
plt.close()

y = np.sin(x/4) * (x**2/200 - x/15 + 3/4) 
plt.plot(x, y)
plt.xlabel("Slab Depth  $[cm]$")
plt.ylabel(r"$J^{+}$")
plt.savefig("hw1q1cplus.pdf")
plt.close()
tot = y
print(y[-1])

y = np.sin(x/4) * (x**2/200 + x/15 + 3/4) 
tot += y
plt.plot(x, y)
plt.xlabel("Slab Depth  $[cm]$")
plt.ylabel(r"$J^{-}$")
plt.savefig("hw1q1cminus.pdf")
plt.close()
print(y[0])

plt.plot(x, tot)
plt.xlabel("Slab Depth  $[cm]$")
plt.ylabel(r"$J$")
plt.savefig("hw1q1d.pdf")
plt.close()