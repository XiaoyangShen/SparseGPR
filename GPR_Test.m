function [ mean, var, MLL ] = GPR_Test( X, y, X_test, lambda, sigma2, sigma02 )
K = SEKernel(X, X, lambda, sigma2);
n = numel(X);
K = K + sigma02 * eye(n);
L = chol(K, 'lower');
alpha = L' \ (L \ y);
MLL = - 0.5 * (y' * alpha) - trace(log(L)) - 0.5 * n * log(2 * pi);

k = SEKernel(X_test', X, lambda, sigma2);
mean = k * alpha;
v = L \ k';
var = diag(SEKernel(X_test', X_test', lambda, sigma2) - v' * v) + sigma02;
end

