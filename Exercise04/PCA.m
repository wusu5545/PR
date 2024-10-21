% Create some random data
D=3;
n=1000;
g_mu=rand(1,D);
g_sigma=cov(rand(4,D));
X = mvnrnd(g_mu,g_sigma,n);

% Plot data cloud
plot3(X(:,1),X(:,2),X(:,3),'.');
hold on

% De-meaning
meanx=mean(X);
meanmat=repmat(meanx,[size(X,1),1]);
Xzero=X-meanmat;

% Compute the SVD of the de-meaned data matrix (which will give you the PCA)
% The transpose is simply because I store the data in a DxN matrix (unlike the Sheet)
[U bla blubb]=svd(Xzero');

% Plot the principal component (red)
plot3(meanx(1),meanx(2),meanx(3), 'go');
a=[meanx+U(:,1)';meanx-U(:,1)'];
plot3(a(:,1),a(:,2),a(:,3),'r-');
% And other base vectors (green, blue)
a=[meanx+U(:,2)';meanx-U(:,2)'];
plot3(a(:,1),a(:,2),a(:,3),'g-');
a=[meanx+U(:,3)';meanx-U(:,3)'];
plot3(a(:,1),a(:,2),a(:,3),'b-');

% Project data into a 2D Space
UL=U(:,1:2)';
Y=Xzero*UL'

% Back-project to 3D (In addition to Y, you will need is the mean and UL)
Xc=Y*UL+meanmat;

% Cmpute mean error
diff=X-Xc;
mean_err=sum(sqrt(sum(diff.^2,2)))/size(diff,1)

% Re-projected points
plot3(Xc(:,1),Xc(:,2),Xc(:,3),'r.');
hold off

% 2D-Plot
figure
plot(Y(:,1), Y(:,2), '.')

