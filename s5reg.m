%% Load data
load('Data_Problem1_regression.mat');
X = [X1 X2]';
Tnew = (9*T1' + 6*T2' + 5*T3' + T4'+ T5') / 22; %r0601195
i = randperm(size(X1, 1), 3000);
X_sample = X(:, i);
T_sample = Tnew(i);


plotSurfaceQ1(X_sample(1,1:1000)',X_sample(2,1:1000)',T_sample(1,1:1000)');


%% Generate performance datastats = grpstats(tbl, {'HiddenCount', 'trainFc', 'transferFc'}, {'median'}, 'DataVars', {'Time', 'MSE_train', 'MSE_test'});

hiddens = [10 20 30 40 50 60 70 80 90 100]; 
repeat_count = 10;trainAlgs = { 'trainlm','trainscg', 'trainbfg', 'trainbr'}; 
transferFcns = {'logsig', 'tansig'};
data = {};

for hidden_count=hiddens
    for trainAlg=trainAlgs
        trainFc = char(trainAlg);
        for transferFcn=transferFcns
            transferFc = char(transferFcn);
            for j=1:repeat_count
                net = feedforwardnet(hidden_count, trainFc);
                net.trainParam.showWindow = false;
                net.divideParam.trainRatio = 1/3;
                net.divideParam.valRatio = 1/3;
                net.divideParam.testRatio = 1/3;
                net.layers{1}.transferFcn = transferFc;
                %output layer transferFcn always purelin for regression%% purposes
                
                tic;
                [net, tr] = train(net, X_sample, T_sample);
                time = toc;
                
                X_train = X_sample(:, tr.trainInd);
                T_train = T_sample(tr.trainInd);
                X_test = X_sample(:, tr.testInd);
                T_test = T_sample(tr.testInd);
                
                T_train_sim = sim(net, X_train);
                T_test_sim = sim(net, X_test);

                mserror_train = mean((T_train - T_train_sim).^2); %== tr.best_perf
                mserror_test = mean((T_test - T_test_sim).^2); %== tr.best_tperf
                
                data{end+1, 1} = hidden_count;
                data{end, 2} = trainFc;
                data{end, 3} = transferFc;
                data{end, 4} = time;
                data{end, 5} = mserror_train;
                data{end, 6} = mserror_test;
            end
        end
    end
end

tbl = cell2table(data, 'VariableNames', {'HiddenCount', 'trainFc', 'transferFc', 'Time', 'MSE_train', 'MSE_test'});
stats = grpstats(tbl, {'HiddenCount', 'trainFc', 'transferFc'}, {'mean'}, 'DataVars', {'Time', 'MSE_train', 'MSE_test'});
stats.Algorithm = strcat(stats.trainFc, '_', stats.transferFc);

