%% Load data
% data = readtable('winequality-red.csv');
data= winequalityred;
c1 = data(data.quality == 5, :);
c2 = data(data.quality == 6, :);
c = [c1; c2];
c.label1 = c.quality == 5;
c.label2 = c.quality == 6;
input = table2array(c);
X = input(:, 1:end-3)';
y = input(:, end-1:end)';

%% Compare network architectures and training functions
hiddens = [5 10 15 20 25 30 35 40 45 50] ; 
trainAlgs = { 'trainlm','trainscg', 'trainbfg'}; 
transferFcns = {'logsig', 'tansig'};
repeat_count = 10;
data = {};

for hidden_count=hiddens
    for trainAlg=trainAlgs
        trainFc = char(trainAlg);
        for transferFcn=transferFcns
            transferFc = char(transferFcn);
            for j=1:repeat_count
                net = patternnet(hidden_count, trainFc);
                net.trainParam.showWindow = false;
                net.layers{1}.transferFcn = transferFc;
                
                tic;
                [net, tr] = train(net, X, y);
                time = toc;
                
                X_val = X(:, tr.valInd);
                T_val = y(:, tr.valInd);

                T_val_sim = sim(net, X_val);
                
                [c,~,~,~] = confusion(T_val, T_val_sim);
                                
                data{end+1, 1} = hidden_count;
                data{end, 2} = trainFc;
                data{end, 3} = transferFc;
                data{end, 4} = time;
                data{end, 5} = 100*(1-c); %CCR
            end
        end
    end
end

tbl = cell2table(data, 'VariableNames', {'HiddenCount', 'trainFc', 'transferFc', 'Time', 'CCR'});
stats = grpstats(tbl, {'HiddenCount','trainFc', 'transferFc'}, {'mean'}, 'DataVars', {'Time', 'CCR'});
stats.Algorithm = strcat(stats.trainFc, '_', stats.transferFc);
 

