function test_targets = NearestNeighbor(train_patterns, train_targets, test_patterns, params)

% Classify using the Nearest neighbor algorithm
% Inputs:
% 	train_patterns	- Train patterns
%	train_targets	- Train targets
%   test_patterns   - Test  patterns
%	params		    - unused
%
% Outputs
%	test_targets	- Predicted targets

L			= length(train_targets);
Uc          = unique(train_targets);

N               = size(test_patterns, 2);
test_targets    = zeros(1,N); 

% Training phase
% Not necessary

% Test phase
for i = 1:N,
    dist            = sum((train_patterns - test_patterns(:,i)*ones(1,L)).^2);
    [d, indices] = sort(dist);
    
    test_targets(i) = train_targets(indices(1));
end