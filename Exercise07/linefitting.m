function linefitting
    
    % Range for x values
    x = 0:0.01:2;
    
    % Generate samples for a line
    p = [0.8 0.25];
    y = p(1)*x + p(2);
    
    % Add some Gaussian noise
    y = y + 0.2 * randn(size(y));
    
    % Add some outlier to the samples
    outlier = (randn(size(y))> 0.9).*randn(size(y))*2.0;
    y = y + outlier;
    y(10) = y(10)+10; y(end-10) = y(end-10) - 10;
    
    % Perform minimization
    p0 = linearEstimate(x, y);
    a = 1e-3; % huber norm parameter
	options = optimset('MaxIter', 1000, 'MaxFunEvals', 1000, 'TolX', 1e-4, 'TolFun', 1e-6, 'GradObj', 'on', 'DerivativeCheck', 'on');
    [p_huber, fval, exitflag, output] = fminunc(@(p)objfcn_huber(p,x,y,a), p0, options)
    
    p
    p0
    p_huber
    % Plot results
    figure; plot(x, y, 'b.'); hold on;
    plot(x, p(1)*x + p(2), 'b'); grid on;
    plot(x, p0(1)*x + p0(2), 'g'); grid on;
    plot(x, p_huber(1)*x + p_huber(2), 'r'); grid on;
    legend('samples','gt', 'least squares', 'huber');
end

% Linear estimate
function p0 = linearEstimate(x, y)
	mx=mean(x);
	my=mean(y);
	mxy=mean(x.*y);
	c=mxy-mx*my;
	a=c/var(x);
	b=my-a*mx;
	p0=[a b];
end % function

% The objective function that should be minimized with respect to the
% line parameters
function [f, g] = objfcn_huber(p, x, y, a)
	% Calculate residual term
	r = y - p(1)*x - p(2);

	% Calculate value for huber norm
	f = sum( huber(r, a) );

	% Calculate gradient
	g(1) = sum(-x .* huber_grad(r, a));    % Derivative w.r.t. p(1)
	g(2) = sum(-huber_grad(r, a));         % Derivative w.r.t. p(2)

	%% Calculate Hessian
	%f_aa = sum(x.^2 .* huber_hessian(r, a));
	%f_bb = sum(huber_hessian(r, a));
	%f_ab = sum(x .* huber_hessian(r, a));
	%H = [f_aa f_ab; f_ab f_bb];
end
	
% Huber loss function
function h = huber(r, a)

    h = zeros(size(r));
    h(abs(r) < a) = r(abs(r) <= a).^2;
    h(abs(r) >= a) = a * (2*abs(r(abs(r) >= a)) - a);
end % function

% Gradient of Huber loss function
function hg = huber_grad(r, a)

    hg = zeros(size(r));
    hg(abs(r) < a) = 2*r(abs(r) < a);
    hg(abs(r) >= a) = 2*a*sign( r(abs(r) >= a) );
end % function

%% Second derivative of Huber loss function
%function h_hessian = huber_hessian(r, a)
%
%    h_hessian = zeros(size(r));
%    h_hessian(abs(r) < a) = 2;
%    h_hessian(abs(r) >= a) = sign( r(abs(r) >= a) );
%       
%end % function
    
        

    