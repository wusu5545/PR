function test_targets = SVM(train_patterns, train_targets, test_patterns, params)

[Dim,Np] = size(train_patterns);
softmargin = params(1);
mu = params(2);

%change classes 0 to -1
train_targets(train_targets == 0) = -1;

% options for fmincon
options = optimset('GradObj','on','MaxFunEvals',1e6, 'MaxIter', 100, 'Display','iter');

if softmargin == 0
    %initialize
    x = zeros(Dim+1, 1);
    % linear inequalities
    % create matrix A, vector b
    A = zeros(Np, Dim + 1);
    b = ones(Np,1);

    for i = 1:Np
        A(i,1:end-1) = (train_patterns(:,i)' .* train_targets(i));
        A(i,end) = (train_targets(i));
    end
    
    % the fmincon function requires: A x <= b !
    A = -A;
    b = -b;
    % Start
    [x_opt,fval,exitflag] = fmincon(@anorm,x,A,b,[],[],[],[],[],options)
else
    x = zeros(Dim+1+Np, 1);
    A = zeros(2*Np, Dim+1+Np);
    b = [ones(Np,1); zeros(Np,1)];
    % linear inequalities for the feature points
    for i = 1:Np
        A(i,1:Dim) = (train_patterns(:,i)' .* train_targets(i));
        A(i,Dim+1) = (train_targets(i));
        A(i,Dim+1+i) = 1;
    end
    % slack variables larger than 0
    A(Np+1:end, Dim+2:end) = eye(Np);
    % the fmincon function requires: A a <= b !
    A = -A;
    b = -b;
    % Start
    anorm_lambda = @(x) anormslack(x, mu, Dim, Np);
    [x_opt,fval,exitflag] = fmincon(anorm_lambda,x,A,b,[],[],[],[],[],options)
end

% value for residual
sum(A*x_opt - b)

% result parameters
x_opt

% Classify test patterns
test_targets = zeros(1, length(test_patterns));
v = x_opt(1:Dim);
v0 = x_opt(Dim+1);

for i = 1:length(test_patterns)
    test_targets(i) = (v' * test_patterns(:,i) + v0) > 0;    
end

end

% Objective function for hard margins
function [res, g] = anorm(a)
    v=a(1:end-1);
    res = 1/2 * norm(v)^2;
    g = [v; 0];
end

% Objective function for soft margins with slack variables
function [res, g] = anormslack(a, mu, Dim, Np)
    v=a(1:Dim);
    xi = a(Dim+2:end);
    res = 1/2 * norm(v)^2 + mu*sum(xi);
    g = [v; 0; mu*ones(Np,1)];
end
