function test_targets = NormNeighbor(train_patterns, train_targets, test_patterns, params)

% Classify using the Norm neighbor algorithm
% The distance to the class means is calculated with different norms.
% Inputs:
% 	train_patterns	- Train patterns
%	train_targets	- Train targets
%   test_patterns   - Test  patterns
%	params		    - which norm to use
%                        0: Mahalonobis
%                        1: L1
%                        2: L2
%
% Outputs
%	test_targets	- Predicted targets

%% Train phase

L			= length(train_targets);
Uc          = unique(train_targets);

% divide trainpatterns into the two classes
idx0 = find(train_targets == 0);
idx1 = find(train_targets == 1);

% train mu and sigma
mu0 = mean(train_patterns(:,idx0),2);
mu1 = mean(train_patterns(:,idx1),2);

sigma0 = cov(train_patterns(:, idx0)');
sigma1 = cov(train_patterns(:, idx1)');

%% Test phase
N               = size(test_patterns, 2);
test_targets    = zeros(1,N); 
dist0 = 0;
dist1 = 0;
for i = 1:N
    % calculate distance
    if params == 1
        dist0 = sum(abs(mu0 - test_patterns(:,i)));
        dist1 = sum(abs(mu1 - test_patterns(:,i)));
    elseif params == 2
        dist0 = sum((mu0 - test_patterns(:,i)).^2);
        dist1 = sum((mu1 - test_patterns(:,i)).^2);
    else
        dist0 = sum((mu0 - test_patterns(:,i))' * inv(sigma0) * (mu0 - test_patterns(:,i)));
        dist1 = sum((mu1 - test_patterns(:,i))' * inv(sigma1) * (mu1 - test_patterns(:,i)));
    end
    
    % do argmax
    if dist0 < dist1
        test_targets(i) = 0;
    else
        test_targets(i) = 1;
    end
end