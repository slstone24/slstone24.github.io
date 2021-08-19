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
---
## [MC Simulated Annealing of Morse Potential Model]


<p style="font-size:11px"></p>
<!-- Remove above link if you don't want to attibute -->
