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

- [MC Simulated Annealing of Morse Potential Model]
This program uses Monte Carlo simulated annealing for optimization. We use the Morse Potential to compute the energy for an arbitrary number of atoms. We then use a Monte Carlo simulation to identify the lowest energy configuration of the atoms.



```MATLAB
clear
close all
% the function Energymorse computes energy for part a)

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
```


For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
- [Project 2 Title](http://example.com/)
- [Project 3 Title](http://example.com/)
- [Project 4 Title](http://example.com/)
- [Project 5 Title](http://example.com/)

---

<p style="font-size:11px"></p>
<!-- Remove above link if you don't want to attibute -->
