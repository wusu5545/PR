function test_targets = kNearestNeighbor(train_patterns, train_targets, test_patterns, kNN)

% Classify using the Nearest neighbor algorithm
% Input:
%   train_patterns  - Train patterns
%   train_targets   - Train targets
%   test_patterns   - Test patterns
%   kNN             - Number of Nearest Neighbors
%
% Output:
%   test_targets    - Predicted targets

L = length(train_targets); % should be the same as number of train_patterns
UC = unique(train_targets) % how many classes are there. UC is a vector!

%just in case...
if(L < kNN),
    error('More neighbors than there are points.')
end

N = size(test_patterns, 2); % How many test patterns do I have?
test_targets = zeros(1,N);  % Create vector that will contain assigned class number

for i = 1:N,
    % calculate euclidean distance between current test_pattern and all
    % train_patterns
    dist            = sum((train_patterns - test_patterns(:,i) * ones(1,L)).^2);
    
    
    % sort according the distance (ascending). indices contains vector with according
    % numbers
    [m, indices]    = sort(dist);
    
    % create histogram of classes: take k nearest neighbors and their
    % classes
    n               = hist(train_targets(indices(1:kNN)), UC);
    
    
    % m contains number of entries, best the index
    [m, best]       = max(n);
    
    % assign to the same class 
    test_targets(i) = UC(best);
end
