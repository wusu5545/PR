function [means, label] = k_means(train_patterns, train_targets, k)

%Reduce the number of data points using the k-means algorithm
%Inputs:
%	train_patterns	- Input means
%	train_targets	- Input targets
%	k				- Number of output data points
%   plot_on         - Plot stages of the algorithm
%
%Outputs
%	means           - New means
%	targets			- New targets
%	label			- The labels given for each of the original means

[dim,Np] = size(train_patterns);
dist	 = zeros(k,Np);
label    = zeros(1,Np);

%Initialize the mu's
mu		= randn(dim,k);
mu		= sqrtm(cov(train_patterns',1))*mu + mean(train_patterns')'*ones(1,k);
old_mu	= zeros(dim,k);

switch k,
case 0,
    mu      = [];
    label   = [];
case 1,
   mu		= mean(train_patterns')';
   label	= ones(1,Np);
otherwise
   while (sum(sum(abs(mu - old_mu) > 1e-5)) > 0),
      old_mu = mu;
      
      %Classify all the means to one of the mu's
      for i = 1:k,
         dist(i,:) = sum((train_patterns - mu(:,i)*ones(1,Np)).^2);
      end
      
      %Label the points
      [m,label] = min(dist);
      
      %Recompute the mu's
      for i = 1:k,
         idx = find(label == i);
         if ~isempty(idx)
            mu(:,i) = mean(train_patterns(:, idx)')';
         end
      end
   end
end
   
means = mu;