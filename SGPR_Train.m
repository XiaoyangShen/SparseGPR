function [ MLL, grad ] = SGPR_Train( X, y, X_sr, lambda, sigma2, sigma02 )
n = numel(X);
m = numel(X_sr);
K_mn = SEKernel(X_sr, X, lambda, sigma2);
K_nm = SEKernel(X, X_sr, lambda, sigma2);
K_mm = SEKernel(X_sr, X_sr, lambda, sigma2);
K = K_nm * pinv(K_mm) * K_mn + sigma02 * eye(n);
Inv_K = pinv(K);
MLL = -0.5 * y' * Inv_K * y - 0.5 * log(det(K)) - 0.5 * n * log(2 * pi); 

% L = chol(K, 'lower');
% alpha = L' \ (L \ y);
% MLL = - 0.5 * (y' * alpha) - trace(log(L)) - 0.5 * n * log(2 * pi);
if nargout > 1 % gradient required
    % grad_lambda = 0.5 * y' * Inv_K * g_lambda * Inv_K * y - 0.5 * trace(Inv_K * g_lambda);
    grad_lambda = 0;
    grad_sigma2 = 0;
    grad = [grad_lambda, grad_sigma2, 0]; 
end
end
