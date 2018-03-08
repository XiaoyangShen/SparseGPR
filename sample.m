rng(1724); % Set a seed for the random function 
load('Kin40k.mat');
% n = 3000;
% ntest = 9000;
% 
% index = randsample(10000, n);
Xtrain = x(:, 1);
Ytrain = y;

index = randsample(30000, 200);
Xtest = xtest(index, 1);
Ytest = ytest(index, :);
save('10k_200sample.mat', 'Xtest', 'Xtrain', 'Ytest', 'Ytrain');