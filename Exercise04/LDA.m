function test_targets = LDA(train_patterns, train_targets, test_patterns, params)
%Classification using the Linear Discriminant Analysis

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training (a)

% Separate classes 
N = length(train_patterns);
classes = unique(train_targets);

if (length(classes) ~= 2) 
    error 'only two-class problems supported';
end

% divide train_patterns into the two classes
idx0 = find(train_targets == 0);
idx1 = find(train_targets == 1);
c0 = train_patterns(:, idx0);
c1 = train_patterns(:, idx1);

% calculate priors
p0 = length(c0)/N
p1 = length(c1)/N

% Calculate class means
mu0 = mean(c0(:,:)')';
mu1 = mean(c1(:,:)')';


% Calculate Joint Covariance 
% Class 0
s0 = zeros(2,2);
for i= 1:length(c0)
    %class 0
    s0 = s0 + (c0(:,i) - mu0)*(c0(:,i) - mu0)';
end

% Class 1
s1 = zeros(2,2);
for i= 1:length(c1)
    %class 1
    s1 = s1 + (c1(:,i) - mu1)*(c1(:,i) - mu1)';
end

% joint covariance matrix
sigma = (s0 + s1)/(length(train_patterns));

%SVD
[U, D] = svd(sigma);
phi = sqrt(inv(D)) * U'


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Classification: (b)

% How many test_targets? Create appropriate test_targets vector
test_targets = zeros(1, length(test_patterns));

%%%%%%%%%%%%%%%%%%%%%%
%   pre-calculate phi(mu_y), and logarithms. 
%   Save some resources!
phiMu0 = phi*mu0;
logP0 = log(p0);

phiMu1 = phi*mu1;
logP1 = log(p1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Choose the class which minimizes given term

for i=1:length(test_patterns)

    y0 = 1/2 * norm(phi*test_patterns(:,i) - phiMu0)^2 - logP0;
    y1 = 1/2 * norm(phi*test_patterns(:,i) - phiMu1)^2 - logP1;
    
    test_targets(i) = y1 < y0;
    
end


end

