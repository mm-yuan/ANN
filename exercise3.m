%% Clean the workspace
clc;
clear;
close all;
%% Set the parameters of the run
n_tr = 300;             % Number of training points (this includes training and validation).
n_te = 200;             % Number of test points
n_neurons = 50;         % Number of neurons
n_neurons2 = 50;         % Number of neurons
n = 1000;               % Total number of samples
ne =3000;               % Number of epochs
perc_training = 0.7;    % Number between 0 and 1. The validation set will be 1-perc_training.

if n < n_tr+n_te
    n = n_tr+n_te;
end

if perc_training >= 1 || perc_training <= 0
    error('The training set is ill defined. The variable perc_training should be between 0 and 1')
end

%% Create the samples
% Allocate memory
u = zeros(1, n);
x = zeros(1, n);
y = zeros(1, n);

% Initialize u, x and y
u(1)=randn; 
x(1)=rand+sin(u(1));
y(1)=x(1);

% Calculate the samples
for i=2:n
    u(i)=randn;
    x(i)=.3*x(i-1)+sin(u(i));
    y(i)=x(i);
end

%% Create the datasets
% Training set
X=num2cell(u(1:n_tr)); 
T=num2cell(y(1:n_tr));

% Test set
T_test=num2cell(y(end-n_te:end)); 
X_test=num2cell(u(end-n_te:end));

%% Train and simulate the network
% Create the net and apply the selected parameters
net = newelm(X,T,n_neurons);        % Create network
%net = newelm(X,T,[n_neurons,n_neurons2]);      
net.trainParam.epochs = ne;         % Number of epochs
net.divideParam.testRatio = 0;
net.divideParam.valRatio = 1-perc_training;
net.divideParam.trainRatio = perc_training;
net.trainParam.goal=0.001;

numNN = 10;
perfs = zeros(1, numNN);
tic;
for i = 1:numNN
  net = train(net,X,T);               % Train network
  T_test_sim = sim(net,X_test);       % Test the network
  perfs(i) = mse(cell2mat(T_test)-cell2mat(T_test_sim));
end
toc;
mean_mse= mean(perfs);


