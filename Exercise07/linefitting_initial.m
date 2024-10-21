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
	
    % TODO: minimization
    
    % Plot results
    figure; plot(x, y, 'b.'); hold on;
    %plot(x, p0(1)*x + p0(2), 'g'); grid on;
    %plot(x, p_huber(1)*x + p_huber(2), 'g'); grid on;
end

% Linear estimate
function p0 = linearEstimate(x, y)

	p0=[a b];
end % function

% The objective function that should be minimized with respect to the
% line parameters
function [f, g] = objfcn_huber(p, x, y, a)


end
	
% Huber loss function
function h = huber(r, a)

        
end % function

% Gradient of Huber loss function
function hg = huber_grad(r, a)

       
end % function

   
    