%% Load data
A = prprob1();
A(A == 0) = -1;
A = [A; A; A];
 
%tmp=reshape(A,[5,7,33]);
%tmp_update=repelem(tmp,2,2);
%A=reshape(tmp_update,[140,33]);

num_letters = size(A, 2);

%% Hopfield - distorted
num_distortions = 3;
nums_iterations = [1 2 5 10 15 20 30 40 50];
Ps = 1:num_letters;
repeat = 10;
error = zeros(numel(nums_iterations), num_letters);
for P=Ps
    for i=1:numel(nums_iterations)
        tmp = zeros(1, repeat);
        for j=1:repeat
            tmp(j) = hop_perf(A, P, nums_iterations(i), num_distortions);
        end
        error(i, P) = mean(tmp);
        if error(i, P) == 0
           error(i+1:end, P) = NaN;
           break;
        end
    end
end



