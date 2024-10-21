function [test_targets, param_struct] = EM(train_patterns, train_targets, test_patterns, Ngaussians)

% Classify using the expectation-maximization algorithm
% Inputs:
% 	train_patterns	- Train patterns
%	train_targets	- Train targets
%   test_patterns   - Test  patterns
%   Ngaussians      - Number for Gaussians for each class (vector)
%
% Outputs
%	test_targets	- Predicted targets
%   param_struct    - A parameter structure containing the parameters of the Gaussians found

classes             = unique(train_targets); %Number of classes in targets
Nclasses            = length(classes);
Dim                 = size(train_patterns,1);

max_iter   			= 100;
max_try             = 5;
Pw					= zeros(Nclasses,max(Ngaussians));
sigma				= zeros(Nclasses,max(Ngaussians),size(train_patterns,1),size(train_patterns,1));
mu					= zeros(Nclasses,max(Ngaussians),size(train_patterns,1));

% The initial guess is based on k-means clustering. We determine means,
% covariance matrices and weights for each GMM component per class.
% If it does not converge after max_iter iterations, a random guess is used.
disp('Using k-means for initial guess')
for i = 1:Nclasses,
    in  			= find(train_targets==classes(i));
    [initial_mu, labels] = k_means(train_patterns(:,in),train_targets(:,in),Ngaussians(i));
    % Determine covariance matrix and weights for GMM in the current class
    for j = 1:Ngaussians(i)
        gauss_labels    = find(labels == j);
        if isempty(gauss_labels)
            Pw(i,j) = 0;
            sigma(i,j,:,:)  = randn(Dim);
        else
            Pw(i,j)         = length(gauss_labels) / length(labels);
            sigma(i,j,:,:)  = diag( std(train_patterns(:,in(gauss_labels))').^2 );
        end
    end
    mu(i,1:Ngaussians(i),:) = initial_mu';
end

% Run EM algorithm using k-means clustering as initial guess. 
for c = 1:Nclasses
    
    % Select patterns from the current class
    train = find(train_targets == classes(c));
    
	% TODO: keep track of EM updates & visualize!!!
    if (Ngaussians(c) == 1),
        % If there is only one Gaussian, there is no need to do a whole 
        % EM procedure. We find mean and covariance matrix using ML
        % estimation.
        sigma(c,1,:,:) = cov(train_patterns(:,train)',1);
        mu(c,1,:) = mean(train_patterns(:,train)');
    else
        sigma_l         = squeeze(sigma(c,:,:,:));
        old_sigma       = zeros(size(sigma_l)); 		%Used for the stopping criterion
        iter			= 0;							%Iteration counter
        n			 	= length(train);				%Number of training points
        qi			    = zeros(Ngaussians(c),n);       %This will hold qi's
        P				= zeros(1,Ngaussians(c));
        Ntry            = 0;
        
        % Iterate E- and M-steps until convergence.
        while ((sum(sum(sum(abs(sigma_l - old_sigma)))) > 1e-4) && (Ntry < max_try))
            
            old_sigma = sigma_l;
            
            % E step: Compute Q(theta; theta_i)
            for j = 1:n
                data  = train_patterns(:,train(j));
                for k = 1:Ngaussians(c)
                    P(k) = Pw(c,k) * p_single(data, squeeze(mu(c,k,:)), squeeze(sigma_l(k,:,:)));
                end          
                
                for l = 1:Ngaussians(c)
                    qi(l,j) = P(l) / sum(P);
                end
            end
            
            % M step: theta_i+1 <- argmax(Q(theta; theta_i))
            % Determine new mean vector
            for l = 1:Ngaussians(c),
                mu(c,l,:) = sum((train_patterns(:,train).*(ones(Dim,1)*qi(l,:)))')/sum(qi(l,:)');
            end
            
            % Determine new covariance matrix.
            for l = 1:Ngaussians(c)
                data_vec = train_patterns(:,train);
                data_vec = data_vec - squeeze(mu(c,l,:)) * ones(1,n);
                data_vec = data_vec .* (ones(Dim,1) * sqrt(qi(l,:)));
                sigma_l(l,:,:) = cov(data_vec',1)*n/sum(qi(l,:)');
            end
            
            
            % Determine new GMM weights.
            Pw(c,1:Ngaussians(c)) = 1/n*sum(qi');
            
            % Remove elements with no assigned sample
            for l = 1:Ngaussians(c)
                if Pw(c,l) == 0
                    disp(['Removing a Gaussian from class: ' num2str(c)])
                    other_idx = 1:Ngaussians(c) ~= l;
                    Ngaussians(c) = Ngaussians(c) - 1;
                    sigma_l(1:Ngaussians(c),:,:) = sigma_l(other_idx,:,:);
                    old_sigma(1:Ngaussians(c),:,:) = old_sigma(other_idx,:,:);
                    mu(c,1:Ngaussians(c),:) = mu(c,other_idx,:);
                    Pw(c,1:Ngaussians(c)) = Pw(c, other_idx);
                    qi = zeros(Ngaussians(c),n);
                    P = zeros(1,Ngaussians(c));
                end
            end
            
            iter = iter + 1;
            disp(['Iteration: ' num2str(iter)])
            
            if (iter > max_iter),
                sigma_l = randn(size(sigma_l));
                iter  = 0;
                Ntry  = Ntry + 1;
                
                if (Ntry > max_try)
                    disp(['Could not converge after ' num2str(Ntry-2) ' redraws. Quitting']);
                else
                    disp('Redrawing weights.')
                end
            end
            
        end
        
        sigma(c,:,:,:) = sigma_l;
    end
end



% Now all parameters for the GMMs in the different classes are available. 
% Classify test patterns using Bayes decision rule
mu  %: mean value
% sigma: covariance matrix
% Pw: prior
prior = zeros(Nclasses);
for c = 1:Nclasses,
    prior(c) = length(find(train_targets == classes(c)))/length(train_targets);
end

% Now all parameters for the GMMs in the different classes are available. 
% Classify test patterns using Bayes decision rule
Ntest = length(test_patterns);
posterior = zeros(Ntest, Nclasses);
for i = 1:Ntest
    for c = 1:Nclasses
        likelihood = 0;
        for k = 1:Ngaussians(c)
            likelihood = likelihood + Pw(c,k) * p_single(test_patterns(:,i), squeeze(mu(c,k,:)), squeeze(sigma(c, k,:,:)));
        end
        posterior(i,c) = prior(c) * likelihood;
    end
    posterior(i,:) = posterior(i,:) / sum(posterior(i,:));  % renormalize to probability
end
    
test_targets = posterior(:,1) < 0.5;


function p = p_single(x, mu, sigma)

if length(mu) == 0
    p = 0;
else
    % Return the probability on a Gaussian probability function. Used by EM
    p = 1/sqrt(det(2*pi*sigma)) *  exp(-0.5 * (x-mu)' * inv(sigma) * (x-mu));
end

