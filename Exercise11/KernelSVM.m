function test_targets = KernelSVM(train_patterns, train_targets, test_patterns, width)

[Dim, Np]       = size(train_patterns);

use_bias = 1;

% get [-1,1] class info
yi = train_targets;
yi(yi == 0) = -1;

% create the kernel matrix from the input patterns using Gaussian RBF
K = zeros(Np); 
for i = 1:Np
    K(:,i) = GaussianRBF(train_patterns(:,i), train_patterns, width);
end

% inequality constraint that lambda_i >= 0 -> can be expressed by lb and ub
A = [];
b = [];

% equality constraint that sum(yi*lambda_i) = 0 
if use_bias
    Aeq = yi;
    beq = 0;
else
    Aeq = [];
    beq = [];
end

% lower and upper bounds on the variables
lb = zeros(1, Np);
ub = [];

% Quadratic programming
options = optimset('Algorithm', 'interior-point-convex', 'Display','iter');
H = diag(yi) * K * diag(yi);
f = -ones(1, Np);

% lambda_star is solution to the dual problem!
lambda_star	= quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options)';
% it is IMPOSSIBLE to compute the solution of the primal problem
% for gaussian kernel, it is an infinite-dimensional vector!

% Find the support vectors
sv_min = 0.0001;
sv = find(lambda_star > sv_min);

% Find the bias using training samples with zero slack
if use_bias
    y_est = (lambda_star.*yi) * K';
    B     = yi(sv) - y_est(sv); % average bias
    bias  = mean(B);
else
    bias = 0;
end

% classify test targets
N = length(test_patterns);
y = zeros(1,N);
for i = 1:length(sv)
    v = lambda_star(sv(i)) * yi(sv(i)) * GaussianRBF(train_patterns(:,sv(i)), test_patterns, width);
    y = y + v';
end

test_targets = y + bias;
test_targets = test_targets > 0;

return

end

% Kernel function
function k = GaussianRBF(x, patterns, width)
    k = 1./ (sqrt(2*pi)*width) .* exp( - sum((patterns - x * ones(1, length(patterns))).^2 )' ./ (2 * width^2) );
end

