function [ fMLL ] = SMLLOpt( X, y, X_sr, hyper)
global LOWER_BOUND
lambda = exp(hyper(1));
sigma2 = exp(hyper(2));
sigma02 = log(1 + LOWER_BOUND + exp(hyper(3)));
[fMLL] = SGPR_Train( X, y, X_sr, lambda, sigma2, sigma02 );
fMLL = - fMLL;
end

