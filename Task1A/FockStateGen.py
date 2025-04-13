"FockStateGen"
import dynamiqs as dq
fock = dq.states.fock
plot_wigner = dq.plot.wigner
import matplotlib.pyplot as plt  

#Fock Sim
state = fock(10, 1) #Initialize Fock state with dimension 10 and 1 quantum
plot_wigner(state)
plt.show() 
