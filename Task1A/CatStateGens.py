"CatStateGens"

import dynamiqs as dq
import numpy as np
coherent = dq.states.coherent
plot_wigner = dq.plot.wigner
import matplotlib.pyplot as plt  

#2-Cat Sim
psi = (dq.coherent(20, 2) + dq.coherent(20, -2)).unit()
dq.plot.wigner(psi)
plt.show()

#3-Cat Sim
psi = (dq.coherent(20, 2) + dq.coherent(20, -2) + dq.coherent(20, 2j)).unit()
dq.plot.wigner(psi)
plt.show()
