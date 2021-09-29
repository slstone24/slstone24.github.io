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



```Matlab
clear
close all
% the function Energymorse computes energy

%initialize variables
Nat = 4;
kT = 0.02;
Niter = 200000;
sigma = 0.1;

r = normrnd(Nat,3);
r = r - ones(Nat,3)*mean(r);

%Metropolis MC Niter times
E = Energymorse(r);
rlist = zeros(Niter,Nat,3); %list of coordinates
Elist = zeros(Niter,1); %list of energies
sigmalist = zeros(Niter,1); %list of sigmas
nacc = 0;

%loop through Niter, store coordinates, energies,
% and respective sigmas
for j=1:Niter
  rnew = r + sigma*rand(size(r));
  rnew = rnew - ones(Nat,1)*mean(rnew);  
  Enew = Energymorse(rnew);
  A = min(1,exp(-(Enew-E)/kT));
  if A > rand 
      % accept
      r = rnew;
      E = Enew;
      nacc = nacc+1;
      sigma = sigma*1.01;
  else
      sigma = sigma*0.99;
  end
  if sigma>1 
      sigma = 1;
  end
  rlist(j,:,:) = r;
  Elist(j) = E;
  sigmalist(j) = sigma;
end
fprintf('kT=%g  accept ratio = %g\n',kT,nacc/Niter);

% show energy history
figure(1)
clf
plot(Elist)
hold on
plot(rlist(:,1,1))
grid
xlabel('step')
legend('E','x1')
title(sprintf('Morse MC Nat=%d kT=%g Niter=%g accratio=%.2f\n',...
    Nat,kT,Niter,nacc/Niter))
set(gca,'fontsize',18)


% plot autocorrelation 
figure(2)
clf
Elist = Elist(2000:end);
xlist = rlist(2000:end,1,1);
EA = fftshift(xcorr(Elist-mean(Elist),'unbiased'));
xA = fftshift(xcorr(xlist-mean(xlist),'unbiased'));
plot(EA(1:3000)/EA(1),'linewidth',2)
hold on
plot(xA(1:3000)/xA(1),'linewidth',2)
grid
xlabel('lag')
ylabel('Auto correlation')
title(sprintf('Morse MC Nat=%d kT=%g Niter=%g accratio=%.2f\n',...
    Nat,kT,Niter,nacc/Niter))
legend('E','x1')
set(gca,'fontsize',18)


% plot the lowest energy config
figure(3)
clf
[Emin,idx]= min(Elist);
fprintf('Emin = %g\n',Emin)
r = squeeze(rlist(idx,:,:));
plot3(r(:,1),r(:,2),r(:,3),'o')
hold on
axis square
grid
title(sprintf('Morse MC Nat=%d kT=%g minimum structure: E=%g',Nat,kT,Emin))
set(gca,'fontsize',18)
for j=1:Nat
    for k=j+1:Nat
        line([r(j,1) r(k,1)],[r(j,2) r(k,2)],[r(j,3) r(k,3)])
    end
end

energy = Energymorse(r);
disp(energy);

function E = Energymorse(r)
  D = 1; % well depth
  a = 1; % well width
  re = 1; % eq bond distance

  % compute energy over grid: sum of Natom pairwise interactions
  Natom = size(r,1);
  E = 0;
      for j=1:Natom
          for k=j+1:Natom
              rjk = norm(r(j,:)-r(k,:));
              E = E + D*(1-exp(-a*(rjk-re)))^2;
          end
      end
end
```
The following plots shows the lowest energy configuration of the atoms and the autocorrelation function, respectively.
<img src="images/MC_config.jpg?raw=true"/>
<img src="images/auto_correlation.jpg?raw=true"/>

---
## PCA Analysis of Fisher's Iris Data
The following Python program shows a PCA (principal component analysis) of Fisher's [Iris dataset](https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv) from scratch. The function `corr(z1, z2)` calculates the desired element of the correlation matrix between two sets of data. The output of this program can be seen below.

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

<p style="font-size:11px"></p>
<!-- Remove above link if you don't want to attibute -->
