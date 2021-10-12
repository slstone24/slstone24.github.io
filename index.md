## Portfolio

---

### Research 

[Computational Bounds to the Quality Factor of a Fabry-Pérot Resonator Through Local Energy Conservation Laws](/pdf/APHY_472_Final_Report.pdf)
<br>
[Power Point Presentation](/pdf/APHY 472 Final Presentation-Final Draft.pdf)
<img src="images/Conservation laws.png?raw=true"/>

---
[On the Accuracy of Coupled Mode Theory With Applications in Integrated Photonic Devices](/pdf/APHY_607_Final_Report.pdf)
<br>
[Power Point Presentation](/pdf/APHY 607 Final Presentation.pdf)
<img src="images/coupled_resonators.png?raw=true"/>

---
[Optimizing Optical Response: Computational Bounds to Quality Factor in 2D Transverse Electric Modes](/pdf/APHY 471 Presentation.pdf)
<img src="images/high_Q_mode.jpg?raw=true"/>

---
### Some Sample Code

---
## PCA Analysis of Fisher's Iris Data
The following Python program shows a PCA (principal component analysis) of Fisher's [Iris dataset](https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv) from scratch. The data istelf shows three species of Iris flowers (Setosa, Virginica, and Versicolor) and the four metrics that describe them: sepal width, sepal length, petal width, and petal length. 
The function `corr(z1, z2)` calculates the desired element of the correlation matrix between two sets of data. The output of this program can be seen below.

```python
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D


def corr(z1, z2):

    n = len(z1)
    z1_bar = np.mean(z1)
    z2_bar = np.mean(z2)
    std1 = np.std(z1)
    std2 = np.std(z2)
    s = 0
    for i, j in zip(z1, z2):
        s = s + ((i-z1_bar) * (j-z2_bar))
    c = s / (n-1)
    r = c / (std1 * std2)
    return r


# open and read csv data into array

with open('iris.csv', newline='') as iris:

    reader = csv.reader(iris)  # read into csv
    X = np.array(list(reader))  # read csv into matrix
    X = np.delete(X, 0, 0)  # delete header row

# flag array for PCA plot
flag = []
for i in range(len(X)):
    if 'Versicolor' in X[i]:
        flag = np.append(flag, "red")
    elif 'Setosa' in X[i]:
        flag = np.append(flag, "blue")
    elif 'Virginica' in X[i]:
        flag = np.append(flag, "green")

X = np.delete(X, -1, -1)  # delete last column
X = X.astype(float)
R = np.ones([X.shape[1], X.shape[1]])
row = R.shape[0]
col = R.shape[1]


for i in range(row):
    for j in range(col):
        R[i][j] = corr(X[:, i], X[:, j])

p, V = LA.eig(R)

## PCA

Y = X@V

# 2 coordinate PCA
plarkart.scatter(Y[:,0], Y[:,1], c=flag, label=flag)
plt.xlabel('Y1')
plt.ylabel('Y2')
plt.title("2-coordinate PCA for Fisher's Iris Data")
plt.show()

# 3 coordinate PCA
ax = plt.axes(projection='3d')
ax.scatter3D(Y[:,0], Y[:,1], Y[:,2], c=flag, label=flag)
ax.set_xlabel('Y1')
ax.set_ylabel('Y2')
ax.set_zlabel('Y3')
plt.title("3-coordinate PCA for Fisher's Iris Data")
plt.show()
```
<img src="images/2d_iris.png?raw=true"/>
<img src="images/3d_iris.png?raw=true"/>

The first image shows the 2D orthogonal subspace that accounts for most of the variance within the data. The axes represent the first two principal components, and the red, blue, and green dots correspond to Versicolor, Setosa, and Virginica Iris flowers, respectively. With this respresentation of the data, we can easily differentiate between the Setosa set and the Virginica/Versicolor set. In order to better differentiate between the Virginica/Versicolor set, we need a third principal component, as seen in the second figure.

---
## Quality Factor Bound of Fabry-Pérot Resonator
The function `bound_Ds` defines an optimization provlem in the form of a quadratically constrained quadratic program (QCQP).

```Matlab
function [cvx_optval,popt] = bound_Ds(S, D, A, epsr, U)
    % optimization problem:
    % max. p'*Im(w'*G0)*p + p'*Im(xi)p
    % s.t. p'*Re{D*S}*p = 0
    %      p'*p = Re(chi)
    % S = G0 + xi_Mat
    % A = Im(w'*G0) + Im(xi)
    % init
    D = D(~cellfun('isempty',D)); % remove empty D entry
    ND = length(D);
    
    % construct matrices for SDP
    C = cell(1,ND);
    for i = 1:ND
        C{i} = U'*Mat_real(D{i}*(S))*U;
    end
    
    % solve SDP by cvx
    A = U'*A*U;
    C{ND+1} = U'*U;
    n = size(A,1);
    
    cvx_begin quiet
        variable X(n,n) hermitian
        maximize( real(A(:)'*X(:)) );
        subject to
            for i = 1:ND
                C{i}(:)'*X(:) == 0;
            end
            C{ND+1}(:)'*X(:) == real(epsr-1);
            X == hermitian_semidefinite(n);
    cvx_end
    
    popt = extract_p_opt(X);
end
```
Upon each iteration *i*, we relax the original quadratic program over semidefinite matrices to arrive at a higher-dimensional linear program, commonnly known as semidefinite relaxation. 
```Matlab
for i = 3:ND
  einc = zeros(size(G0,1),1);
  D{i} = get_Dopt(popt{i-1},S,einc);
  [fmax(i),popt{i}] = bound_Ds(S,D,A,epsr,U);
  fprintf(’ D matrix: %d / %d, fmax = %s \n’, i, ND, fmax(i))
end
```
The optimal polarization current is calculated as the largest eigenvector from the previous iteration of semidefintie relaxation: `[p_opt,~] = eigs(X,1)`. We use this eigenvector to compute the "most violated" D matrix constraint to impose upon the next iteration of our convex optimization:
```Matlab
function Dopt = get_Dopt(p0,S,einc)
    
    if isnan(p0) % if optimal current is nan, return a random diagonal D matrix
        Dopt = 1j*diag(rand(size(einc)));
        fprintf('add a random D matrix as Dopt ... \n')
        return
    end
        
    % diagonal D matrix
    q = S*p0 + einc;
    d = p0.*conj(q);
    d = d/max(d);
    Dopt = diag(d); % maximally violated constraint
end
```
The final results are shown below.
<img src="images/results_final.jpg?raw=true"/>

---
## MC Simulated Annealing of Morse Potential Model
This program uses Monte Carlo simulated annealing for optimization. We use the Morse Potential to compute the energy for an arbitrary number of atoms. We then use a Monte Carlo simulation to identify the lowest energy configuration of the atoms.



```python
import numpy as np
from numpy.linalg import norm
import math
import matplotlib.pyplot as plt
from scipy import signal


def energy_morse(r):
    """
    Given spatial coordinates from Nat atoms,
    return energy according to Morse potential model
    """
    D = 1
    a = 1
    re = 1

    Natom = np.size(r, 0)
    E = 0
    for j in range(Natom):
        for k in range(j+1, Natom):
            rjk = norm(r[j, :] - r[k, :])
            E = E + D * (1 - math.exp(-a * (rjk - re))) ** 2

    return E

# initialize variables
Nat = 4
kT = 0.02
Niter = 200000
sigma = 0.1

# initialize matrices to store energies and coordinates
r = np.random.normal(Nat, 3)
r = r - np.ones((Nat, 3)) * np.mean(r)
E = energy_morse(r)
rlist = np.zeros((Niter, Nat, 3))
Elist = np.zeros((Niter, 1))
sigmalist = np.zeros((Niter, 1))
nacc = 0

# loop through Niter times, store coordinates, energies, and sigmas
for j in range(Niter):
    rnew = r + sigma * np.random.uniform(size=r.shape)
    rnew = rnew - np.ones((Nat, 1)) * rnew.mean(axis=0)
    Enew = energy_morse(rnew)
    A = np.minimum(1, math.exp(-(Enew - E) / kT))

    # if A > u, with u as uniform
    # random deviate, accept, otherwise
    # reject
    if A > np.random.uniform():
        r = rnew
        E = Enew
        nacc += 1
        sigma = sigma * 1.01
    else:
        sigma = sigma * 0.99
    if sigma > 1:
        sigma = 1

    rlist[j, :, :] = r
    Elist[j] = E
    sigmalist[j] = sigma

# plot energy history
print('kt =', kT, 'accept ratio =', nacc / Niter)
plt.plot(Elist)
plt.plot(rlist[:, 0, 0])
plt.xlabel('Step')
plt.legend(['Energy', 'x1'])
plt.title('Energy History, Natom = {Natom}, kT = {kT}'.format(Natom=Nat, kT=kT))
plt.show()

# plot autocorrelation
Elist = Elist[2000:-1]
xlist = rlist[2000:-1, 0, 0]
EA = np.fft.fftshift(signal.correlate(Elist-np.mean(Elist), Elist-np.mean(Elist)))
xA = np.fft.fftshift(signal.correlate(xlist-np.mean(xlist), xlist-np.mean(xlist)))
plt.plot(EA[0:3000]/EA[0])
plt.plot(xA[0:3000]/xA[0])
plt.legend(['Energy', 'x1'])
plt.title('Auto-correlation for Morse MC')
plt.xlabel('lag')
plt.ylabel('Auto-correlation')
plt.show()

# plot energy config
Emin = min(Elist)
idx = np.argmin(Elist)
r = np.squeeze(rlist[idx, :, :])
print('Emin =', Emin)
print('Energy =', energy_morse(r))
ax = plt.axes(projection='3d')
ax.scatter3D(r[:, 0], r[:, 1], r[:, 2], s=40)
for j in range(Nat):
    for k in range(1, Nat):
        plt.plot([r[j, 0], r[k, 0]], [r[j, 1], r[k, 1]], [r[j, 2], r[k, 2]],
                 color='black', markersize=0.005)
plt.title('Minimum Energy Structure, Natom = {Natom}, kT = {kT}'.format(Natom=Nat, kT=kT))
plt.show()
```

The following plots shows the lowest energy configuration of the atoms, autocorrelation function of the energies and their spatial positions, and the energy and spatial coordiante histories, respectively.
<img src="images/MC_config.jpg?raw=true"/>
<img src="images/auto_correlation.jpg?raw=true"/>

<p style="font-size:11px"></p>
<!-- Remove above link if you don't want to attibute -->
