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

## MC Simulated Annealing of Morse Potential Model
This program uses Monte Carlo simulated annealing for optimization. We use the Morse Potential to compute the energy for an arbitrary number of atoms. We then use a Monte Carlo simulation to identify the lowest energy configuration of the atoms.



```javascript
if (isAwesome){
  return true
}
```

---
## Quality Factor Bound of Fabry-Pérot Resonator

```MATLAB
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


---

<p style="font-size:11px"></p>
<!-- Remove above link if you don't want to attibute -->
