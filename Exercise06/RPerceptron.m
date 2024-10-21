function test_targets = RPerceptron(train_patterns, train_targets, test_patterns, max_iter)
% Classify using Rosenblatt's Perceptron algorithm
% Inputs:
% 	train_patterns	- Train patterns
%	train_targets	- Train targets
%   test_patterns   - Test  patterns
%	alg_param	    - Number of iterations
%
% Outputs
%	test_targets	- Predicted targets
%

% number of classes
Nc = length(unique(train_targets));

% number of training patterns
Np = length(train_targets);

% pattern dimensionality
D = size(train_patterns, 1);

% provide initialization
a = rand(D,1);
a0 = 0;
iter = 0;

% learning rate
learningRate = 1;

% compute ground truth y values: [0,1] -> [-1,1]
y = zeros(1, Np);
y(train_targets == 0) = -1;
y(train_targets == 1) = 1;

% Stochastic or normal gradient descent
batch = false;

while (iter < max_iter)
    % index of sampled misclassified feature
    mc = [];
    
    % Randomly sample (without replacement) from the training set 
    % to find missclassified samples
    for k = randperm(Np)
        p = train_patterns(:, k);
        d = a' * p + a0;
        
        % check if misclassified
        if (sign(d) ~= y(k))
            mc = [mc, k];
            if ~batch
                break;
            end
        end
    end

    % Check for convergence
    if (isempty(mc))
        break;
    end
    
    % update with gradient
    g = zeros(D+1, 1);
    for i=1:length(mc)
        g(1:D) = g(1:D) - y(mc(i)) * train_patterns(:, mc(i));
        g(D+1) = g(D+1) - y(mc(i));
    end
    
    % normalize gradient in order to remain stable optimization
    %g = g ./ norm(g);
    
    % gradient always points to the direction of the ascent, in order
    % to get a minimization of the objective function, we have to step into
    % the negative gradient direction (descent)
    a = a - learningRate * g(1:D);
    %a = a ./ norm(a); 
    a0 = a0 - learningRate * g(D+1);
    
    iter = iter + 1;
end

disp(['Final number of iterations: ' num2str(iter)]);

% classify test patterns
test_targets = sign(a' * test_patterns(:,:) + a0);
test_targets(test_targets > 0) = 1;
test_targets(test_targets < 0) = 0;
