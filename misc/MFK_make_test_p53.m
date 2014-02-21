function [trajectories]=MFK_make_test_p53(out_filename)
%%
% time span
tmin=0;
tmax=40;
tspan=[tmin tmax];
%figure;
nvariables=3;

% parameters
Omega=1;

k1=90;
k2=0.002;
k3=1.7;
k4=1.1;
k5=0.93;
k6=0.96;
k7=0.01;
xt=50;
x1=70;
x2=30;
x3=60;

init_val=[zeros(19,1)];
%1, 5, 9, 14, 20, 27
%3, 9, 19, 34, 55, 83
init_val(1)=x1;
init_val(2)=x2;
init_val(3)=x3;

nMoments=3;
multivariate=true;
model_name='p53'
parameters=[k1 k2 k3 k4 k5 k6 k7]
closure='log-normal'
par=[k1 k2 k3 k4 k5 k6 k7 Omega xt nvariables nMoments];


tic
% evaluation
clear mysolution
%all_equations=str2func(['MFK_equations_p53_2mom']);
%all_equations=str2func(['MFK_equations_p53_3mom']);
% all_equations=str2func(['MFK_equations_p53_4mom']);
% all_equations=str2func(['MFK_equations_p53_6mom']);
% all_equations=str2func(['MFK_equations_p53_2momgamma']);
% all_equations=str2func(['MFK_equations_p53_2momgamma1']);
% all_equations=str2func(['MFK_equations_p53_2momgamma2']);
% all_equations=str2func(['MFK_equations_p53_3momgamma']);
% all_equations=str2func(['MFK_equations_p53_3momgamma1']);
% all_equations=str2func(['MFK_equations_p53_3momgamma2']);
% all_equations=str2func(['MFK_equations_p53_4momgamma']);
% all_equations=str2func(['MFK_equations_p53_4momgamma1']);
% all_equations=str2func(['MFK_equations_p53_4momgamma2']);
% all_equations=str2func(['MFK_equations_p53_5momgamma']);
% all_equations=str2func(['MFK_equations_p53_5momgamma1']);     
% all_equations=str2func(['MFK_equations_p53_5momgamma2']);
% all_equations=str2func(['MFK_equations_p53_2momlogn']);
% all_equations=str2func(['MFK_equations_p53_2momlogn0']);
 all_equations=str2func(['MFK_equations_p53_3momlogn']);
%all_equations=[MFK_create_symbolic_automatic_lognormal(nMoments, multi)]
% all_equations=str2func(['MFK_equations_p53_3momlogn0']);
% all_equations=str2func(['MFK_equations_p53_4momlogn']);
% all_equations=str2func(['MFK_equations_p53_4momlogn0']);
%all_equations=str2func(['MFK_equations_p53_5momlogn']);
%all_equations=str2func(['MFK_equations_p53_5momlogn0']);
%all_equations=str2func(['MFK_equations_p53_2momgauss']);
%all_equations=str2func(['MFK_equations_p53_3momgauss']);
options=odeset('MaxStep',0.01);
mysolution=ode15s(all_equations,tspan,init_val,options,par);
timepoints=tmin:0.05:tmax;
trajectories=deval(mysolution,timepoints);
toc
save(out_filename);
%plot(x, traj(1, :), x, traj(2, :), x, traj(3,:))



 




