from numpy import loadtxt
import numpy as np
from scipy.stats import wilcoxon


#acc_reg = loadtxt("./7result_True_0.1.txt", comments="#", delimiter=",", unpack=False)


#acc_reg = loadtxt("./8result_kb_True_0.1.txt", comments="#", delimiter=",", unpack=False)
#acc = loadtxt("./8result_kb_False_0.1.txt", comments="#", delimiter=",", unpack=False)

#acc_reg = loadtxt("./10result_kb_True_0.1.txt", comments="#", delimiter=",", unpack=False)
#acc = loadtxt("./10result_kb_False_0.1.txt", comments="#", delimiter=",", unpack=False)

#acc_reg = loadtxt("./14result_chest_True_0.15.txt", comments="#", delimiter=",", unpack=False)

acc_reg = loadtxt("./18result_chest_True_7.0.txt", comments="#", delimiter=",", unpack=False)
acc = loadtxt("./17result_chest_False_0.1.txt", comments="#", delimiter=",", unpack=False)

print("acc_reg {}, acc {}".format(np.mean(acc_reg), np.mean(acc)))
n = np.min([len(acc_reg), len(acc)])
print(n)
d = acc_reg[:n] - acc[:n]
print("acc_reg")
print(acc_reg)
print("acc")
print(acc)
w, p = wilcoxon(acc_reg[:n], acc[:n], alternative="greater")
print("w, p = {} {}".format(w,p))
breakpoint()