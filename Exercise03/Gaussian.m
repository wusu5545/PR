function test_targets = Gaussian(train_patterns, train_targets, test_patterns, params)
% Classification with logistic regression, one
% Gaussian for each class

% =====================Training Step===================================
% get priors, mu's and sigma's from the train_patterns and
% train_targets

N = length(train_patterns);
classes = unique(train_targets);

if (length(classes) ~= 2) 
    error 'only two-class problems supported';
end

% divide testpatterns into the two classes
idx1 = find(train_targets == 0);
idx2 = find(train_targets == 1);

% train mu and sigma
mu1 = mean(train_patterns(:, idx1)')'
mu2 = mean(train_patterns(:, idx2)')'

sigma1 = cov(train_patterns(:, idx1)')
sigma2 = cov(train_patterns(:, idx2)')

% calculate priors
p1 = length(idx1) / N;
p2 = length(idx2) / N;

N_test = length(test_patterns);
test_targets = zeros(1, N_test);

% =====================Classification Step===================================

% get inverses of sigma-matrices
S1_inv = inv(sigma1);
S2_inv = inv(sigma2);


% calculate constant term
t0 = log(p1/p2);

t1 = log(det((2*pi*sigma2)) ./ det((2*pi*sigma1)));
t2 = (mu2' * S2_inv * mu2) - (mu1' * S1_inv * mu1);

a0 = t0 + 1/2 * (t1 + t2);

% calculate vector for linear term
at = mu1' * S1_inv - mu2' * S2_inv;

% calculate matrix for quadratic term
A = 1/2 * (S2_inv - S1_inv);

% classify patterns
for k = 1:N_test
    p0 = 1.0 / (1 + exp(-(test_patterns(:,k)' * A * test_patterns(:,k) + at * test_patterns(:,k) + a0)));
    if (p0 > 0.5)
        test_targets(1,k) = 0;
    else
        test_targets(1,k) = 1;
    end
end

end
