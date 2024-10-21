function test_targets = Naive_Bayes(train_patterns, train_targets, test_patterns, params)
% Naive Bayes:  Classification with Naive Bayes, assuming independency
% between dimensions
% Gaussian for each class
% 

% =====================Training Step===================================
% get priors, mu's and sigma's from the train_patterns and
% train_targets

classes = unique(train_targets);

if (length(classes) ~= 2)
    error 'only two-class problems supported';
end

% divide testpatterns into the two classes
idx0 = find(train_targets == 0);
idx1 = find(train_targets == 1);

% train mu and sigma
mu11 = mean(train_patterns(1,idx0));
mu12 = mean(train_patterns(2,idx0));

mu21 = mean(train_patterns(1,idx1));
mu22 = mean(train_patterns(2,idx1));


% for the covariances, only consider diagonal elements
sigma1 = zeros(2,2);
sigma1(1,1) = cov(train_patterns(1, idx0)');
sigma1(2,2) = cov(train_patterns(2, idx0)');


sigma2 = zeros(2,2);
sigma2(1,1) = cov(train_patterns(1, idx1)');
sigma2(2,2) = cov(train_patterns(2, idx1)');


% calculate priors
p1 = length(idx0) / length(train_patterns);
p2 = length(idx1) / length(train_patterns);

N = length(test_patterns);
test_targets = zeros(1, N);

N

% =====================Classification Step===================================

for k = 1:N
    % use 1-D Gaussians for the probabilities of the feature components
    prob1 = Gaussian(test_patterns(1,k),mu11,sigma1(1,1));
    prob2 = Gaussian(test_patterns(2,k),mu12,sigma1(2,2));
       
    t1 = p1 * prob1 * prob2;
  
    
    prob1 = Gaussian(test_patterns(1,k),mu21,sigma2(1,1));
    prob2 = Gaussian(test_patterns(2,k),mu22,sigma2(2,2));
    
    t2 = p2 * prob1 * prob2;
    
    % decision rule: take the class with the higher p * G1 * G2
    test_targets(1,k) = t2 > t1;
end

end



% evaluate Gaussian distribution
function g = Gaussian(x, mu, sigma)

d = sqrt(det(2*pi*sigma));
if (d == 0)
    error 'singular covariance matrix'
end
   
g = 1.0 / d * exp(-0.5 * (x - mu)' * inv(sigma) * (x - mu));

end