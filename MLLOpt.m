function [ fMLL, grad ] = MLLOpt( X, y, hyper)
global LOWER_BOUND
lambda = exp(hyper(1));
sigma2 = exp(hyper(2));
sigma02 = log(1 + LOWER_BOUND + exp(hyper(3)));
[fMLL, grad] = GPR_Train( X, y, lambda, sigma2, sigma02 );
fMLL = - fMLL;
grad = - grad;
end

