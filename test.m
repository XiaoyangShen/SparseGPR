clear all; close all;
global LOWER_BOUND
LOWER_BOUND = 0.0001;
lambda = 0.5;
sigma2 = 0.1;
sigma02 = 0.1;

% rng('shuffle')
% noise=0.1^2;                                   % gaussian noise
% N=10000;                                        % dataset size
% N1=201;                                         % trainset size; should be odd integer
% X_test=linspace(0,1,N);
% y0=sqrt(noise)*randn(N,1);
% X=linspace(0,1,N1);
% y=sqrt(noise)*randn(N1,1);
% y=y+arrayfun(@(x) x.*sin(20*x), X)';
% X=X';
load('10k_200sample.mat')
X = Xtrain;
y = Ytrain;
X_test = Xtest;
X_sr = X(1:10:end, :);
method = 2;

% noise = 0.1;
% method = 1;
% training_size = 100;
% testing_size = 1001;
% step = 20;
% X = rand(training_size, 1)*10;
% y = sin(X) + sqrt(noise)*randn(training_size, 1);
% X_test = linspace(0, 10, testing_size);
% X_sr = X(1:step:end);
% 
% X = 10 * rand(150, 1) - 5;
% y = heaviside(X);
% X_test = linspace(-5, 5, 1000);
% X_sr = X(1:step:end);

% [mean, var, ~] = SGPR_Test(X, y, X_sr, X_test, lambda, sigma2, sigma02);
% plot(X_test, mean);
% error = sqrt(var);
% fill([X_test'; flipud(X_test')], [(mean + error); flipud((mean - error))], 'r', 'edgecolor', 'none');
% alpha(0.3);
if method == 1
    tic;
    fun = @(hyper)SMLLOpt( X, y, X_sr, hyper);
    hyper0 = [0, 1, -1];
    lambda = exp(hyper0(1));
    sigma2 = exp(hyper0(2));
    sigma02 = log(1 + LOWER_BOUND + exp(hyper0(3)));
    opthyper = fminunc(fun, hyper0);
    lambda = exp(opthyper(1));
    sigma2 = exp(opthyper(2));
    sigma02 = log(1 + LOWER_BOUND + exp(opthyper(3)));
    [mean1, var1, MLL] = SGPR_Test(X, y, X_sr, X_test, lambda, sigma2, sigma02);
    toc
    fprintf("MLL: %f", MLL);
% else
    tic;
    fun = @(hyper)MLLOpt( X, y, hyper);
    hyper0 = [1, 1, 1];
    opthyper = fminunc(fun, hyper0);
    lambda = exp(opthyper(1));
    sigma2 = exp(opthyper(2));
    sigma02 = log(1 + LOWER_BOUND + exp(opthyper(3)));
    [mean2, var2, MLL] = GPR_Test(X, y, X_test, lambda, sigma2, sigma02);
    toc
    fprintf("MLL: %f", MLL);
    tic;
    options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
    fun = @(hyper)MLLOpt( X, y, hyper);
    hyper0 = [1, 1, 1];
    opthyper = fminunc(fun, hyper0, options);
    lambda = exp(opthyper(1));
    sigma2 = exp(opthyper(2));
    sigma02 = log(1 + LOWER_BOUND + exp(opthyper(3)));
    [mean3, var3, MLL] = GPR_Test(X, y, X_test, lambda, sigma2, sigma02);
    toc
    fprintf("MLL: %f", MLL);
else
    lambda = 1;
    sigma2 = 1;
    sigma02 = 1;
    tic;
    SGPR_Train(X, y, X_sr, lambda, sigma2, sigma02)
    toc
    tic;
    SGPR_Test(X, y, X_sr, X_test, lambda, sigma2, sigma02);
    toc
    tic;
    GPR_Train(X, y, lambda, sigma2, sigma02)
    toc
    tic;
    GPR_Test(X, y, X_test, lambda, sigma2, sigma02);
    toc
end


% hfig=figure('position',[50 100 1800 600]); set(hfig,'Color','w');
% 
% subplot(1,3,1);
% plot(X, y, '+'); hold on;
% plot(X_sr, zeros(numel(X_sr)), 'o');
% plot(X_test, mean1);
% error1 = sqrt(var1);
% fill([X_test'; flipud(X_test')], [(mean1 + error1); flipud((mean1 - error1))], 'r', 'edgecolor', 'none');
% alpha(0.3);
% title('Sparse GP')
% 
% subplot(1,3,2);
% plot(X, y, '+'); hold on;
% plot(X_test, mean2);
% error2 = sqrt(var2);
% fill([X_test'; flipud(X_test')], [(mean2 + error2); flipud((mean2 - error2))], 'r', 'edgecolor', 'none');
% alpha(0.3);
% title('Standard GP, without gradient')

% subplot(1,3,3);
% plot(X, y, '+'); hold on;
% plot(X_test, mean3);
% error3 = sqrt(var3);
% fill([X_test'; flipud(X_test')], [(mean3 + error3); flipud((mean3 - error3))], 'r', 'edgecolor', 'none');
% alpha(0.3);
% title('Standard GP, with gradient')
