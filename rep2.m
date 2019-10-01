%%%%%%%%%%%
% rep2.m
% A script which generates n random initial points
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%
clear all
close all
T = [1 1;-1 -1; 1,-1]';
net = newhop(T);
plot(T(1,:),T(2,:),'gO');
hold on
n=10;
for i=1:n
    a={rands(2,1)};  % generate an initial point
    [y,Pf,Af] = net({50},{},a);   % simulation of the network for 50 timesteps
    record=[cell2mat(a) cell2mat(y)];   % formatting results
    start=cell2mat(a);                  % formatting results
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,50),record(2,50),'gO');  % plot the final point with a green circle
end
legend('attractor','initial state','time evolution','Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');
