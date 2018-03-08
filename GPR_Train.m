function [ MLL, grad ] = GPR_Train( X, y, lambda, sigma2, sigma02 )
[K, g_lambda, g_sigma2] = SEKernel(X, X, lambda, sigma2);
n = numel(X);
K = K + sigma02 * eye(n);
% L = chol(K, 'lower');
% alpha = L' \ (L \ y);
% MLL = - 0.5 * (y' * alpha) - trace(log(L)) - 0.5 * n * log(2 * pi);
Inv_K = pinv(K);
MLL = -0.5 * y' * Inv_K * y - 0.5 * log(det(K)) - 0.5 * n * log(2 * pi); 
if nargout > 1 % gradient required
    grad_lambda = 0.5 * y' * Inv_K * g_lambda * Inv_K * y - 0.5 * trace(Inv_K * g_lambda);
    grad_sigma2 = 0.5 * y' * Inv_K * g_sigma2 * Inv_K * y - 0.5 * trace(Inv_K * g_sigma2); 
    grad_sigma02 = 0.5 * y' * Inv_K * eye(n) * Inv_K * y - 0.5 * trace(Inv_K * eye(n)); 
    grad = [grad_lambda, grad_sigma2, grad_sigma02]; 
end
end
