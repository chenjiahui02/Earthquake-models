import numpy as np
import matplotlib.pyplot as plt

# likelihood of homogeneous Poisson process
def likelikhood(lamb, tl, n):
    return np.exp(-lamb*tl)*lamb**n

# first prior: N(a,b)
def prior(x,a,b):
    return np.exp(-0.5*(x-a)**2/b)/np.sqrt(2*np.pi*b)
# second prior: uniform(c,d)
def prior2(x, c, d):
    if c< x <d:
        return 1/(d-c)
    else:
        return 0
    
# proposal q(x|y) follows N(y, 1)
def proposal(x, y):
    return np.exp(-0.5*(x-y)**2)/np.sqrt(2 * np.pi)

x0 = 5

tl= 50
n=79

# set prior to be 5,5
a=5
b=5

c=0
d=10

sample_list = []

for i in range(5000):
    x1 = np.random.normal(x0,1)
    u = np.random.uniform()
    acc = likelikhood(x1,tl,n)*prior(x1,a,b)*proposal(x0, x1) / (likelikhood(x0,tl,n)*prior(x0,a,b)*proposal(x1, x0))
    if u<acc:
        sample_list.append(x1)
        x0=x1
    else:
        sample_list.append(x0)

        
sample_list2 = []

for i in range(5000):
    x1 = np.random.normal(x0,1)
    u = np.random.uniform()
    acc = likelikhood(x1,tl,n)*prior2(x1,c,d)*proposal(x0, x1) / (likelikhood(x0,tl,n)*prior2(x0,c,d)*proposal(x1, x0))
    if u<acc:
        sample_list2.append(x1)
        x0=x1
    else:
        sample_list2.append(x0)

# plot 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,4))
ax1.hist(sample_list[-1000:], alpha=0.8, bins=50, label="normal prior")
ax1.axvline(1.5, color = "red", alpha=0.8, label="true intensity")
ax1.axvline(np.mean(sample_list[-1000:]), color = "orange", alpha=0.8, label="true intensity")
ax2.hist(sample_list2[-1000:], alpha=0.8, bins=50, label="uniform prior")
ax2.axvline(1.5, color = "red",alpha=0.8, label="true intensity")
ax2.axvline(np.mean(sample_list2[-1000:]), color = "orange",alpha=0.8, label="true intensity")
ax1.legend()
ax1.set_title("a")
ax2.legend()
ax2.set_title("b")
plt.show()