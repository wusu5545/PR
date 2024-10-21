% Create some random data
D=3;
n=1000;
g_mu=rand(1,D);
g_sigma=cov(rand(4,D));
X = mvnrnd(g_mu,g_sigma,n);

% Plot data cloud
plot3(X(:,1),X(:,2),X(:,3),'.');
