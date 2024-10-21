function test_targets = LinearLogisticRegression(train_patterns, train_targets, test_patterns, params)
% Logistic Regression with linear decision boundary.

N = length(train_targets);  % Number of training samples
D = size(train_patterns, 1) + 1; % Dimensionality of feature space

% theta: parameter value for optimization
curTheta = rand(D,1); % initialize with random numbers
oldTheta = curTheta;
converged = 0;

% trace the objective function
vecL = [];

% Training step
iter = 0;
while(converged ~= 1)
    % initialization
 	gradL = zeros(D,1);
    H = zeros(D,D);
    L = 0;
    
    % Sum contributions of all training samples
    for i=1:N
        x = [train_patterns(:,i); 1];
        y = train_targets(i);
        g = gFunc(x, curTheta);
        
        % evaluate objective function
        if (g > 0 && g < 1)
            L = L + (y * log(g) + (1-y)*log(1-g));
        end
        
        %calculate gradient
        faktor = y - g;
        gradL = gradL + faktor.* x;
        
        %calculate Hessian matrix
        faktorH = - g * (1-g);
        H = H + faktorH .* x * x';
    end
    gradL
    H
    
    % store the objective function values for visualization
    vecL = [vecL, L];
    
    % we are maximizing, so H has to be negative definite
    e = eig(H)
    if max(e) >= 0
        converged = 1;
    end

    % limit the number of iterations
    if iter > 100
        converged = 1;
    end

    oldTheta = curTheta;

    if converged ~= 1
        % apply Newton-Raphson parameter update
        curTheta = curTheta - inv(H)*gradL;

        % check for convergence
        test = norm(curTheta - oldTheta);
        if(test < 0.0001)
            converged = 1;
        end

        iter=iter+1; 
    end
end

curTheta
test = norm(curTheta - oldTheta);
iter

% Classification step
test_targets = zeros(1,length(test_patterns));
for i=1:length(test_patterns)
    x = [test_patterns(:,i); 1];
    test_targets(i) = gFunc(x, curTheta) > 0.5;
end


% plot objective function
figure(2);
plot(1:length(vecL), vecL);
title('Log-Likelihood');
xlabel('Iteration');
ylabel('L');
figure(1);

end


function g = gFunc(x, theta)
    p = theta' * x;
    g = 1/(1+ exp(-p)); 
end
        
        
        
        
        
        