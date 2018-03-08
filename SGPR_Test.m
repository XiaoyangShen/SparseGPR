function [ mean, var, MLL ] = SGPR_Test( X, y, X_sr, X_test, lambda, sigma2, sigma02 )
n = numel(X);
m = numel(X_sr);
K_mn = SEKernel(X_sr, X, lambda, sigma2);
K_nm = SEKernel(X, X_sr, lambda, sigma2);
K_mm = SEKernel(X_sr, X_sr, lambda, sigma2);
K = K_mn * K_nm + sigma02 * K_mm;
Inv_K = pinv(K);
k = SEKernel(X_test', X_sr, lambda, sigma2);
mean = k * Inv_K * K_mn * y;
var = diag(sigma02 * k * Inv_K * k') + sigma02;
MLL = 0;
% MLL = -0.5 * y' * Inv_K * y - 0.5 * log(det(K)) - 0.5 * n * log(2 * pi); 
end

