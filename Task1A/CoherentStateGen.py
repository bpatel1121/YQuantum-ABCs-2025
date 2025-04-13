"CoherentStateGen"

import dynamiqs as dq
import numpy as np
coherent = dq.states.coherent
plot_wigner = dq.plot.wigner
import matplotlib.pyplot as plt  


#Coherent Sim
alpha = 2.0
state = coherent(10, alpha)
plot_wigner(state)
plt.show()
