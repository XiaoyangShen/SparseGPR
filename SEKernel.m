function [ K, g_lambda, g_sigma2 ] = SEKernel( rec, src, lambda, sigma2 )
n = size(rec);
n = n(1);
m = size(src);
m = m(1);
K = ones(n, m);
[S1, S2] = ndgrid(rec(:, 1), src(:, 1));
K = sigma2 .* exp(-(S1 - S2) .^2 / lambda);
if nargout > 1 % gradient required
    g_lambda = K .* exp((S1 - S2) .^2 / lambda ^2);
    g_sigma2 = K ./ sigma2;
end
end

