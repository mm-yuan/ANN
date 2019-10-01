%% final model: trainlm logsig 20

net = feedforwardnet(20, 'trainlm');
net.divideParam.trainRatio = 1/3;
net.divideParam.valRatio = 1/3;
net.divideParam.testRatio = 1/3;
[net, tr] = train(net, X_sample, T_sample);

X_train = X_sample(:, tr.trainInd);
T_train = T_sample(tr.trainInd);
X_test = X_sample(:, tr.testInd);
T_test = T_sample(tr.testInd);


T_train_sim = sim(net, X_train);
T_test_sim = sim(net, X_test);

figure;
subplot(121);
plotSurfaceQ1(X_test(1,:)', X_test(2,:)', T_test');
subplot(122);
plotSurfaceQ1(X_test(1,:)', X_test(2,:)', T_test_sim');


mserror_train = mean((T_train - T_train_sim).^2) %== tr.best_perf
mserror_test = mean((T_test - T_test_sim).^2) %== tr.best_tperf
          
          
