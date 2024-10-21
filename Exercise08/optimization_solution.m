function [] = optimization()

    close all
    
    [X1, X2] = meshgrid(-2:0.05:1, -1:0.05:1);
    
    y = f([X1(:)'; X2(:)']);
    y = reshape(y, size(X1, 1), size(X2, 2));
    
    figure(1)
    
    surfc(X1, X2, y, 'FaceColor','red','EdgeColor','none');
    camlight left; lighting phong
    set(gca, 'GridLineStyle', '--');
    xlabel('x_1');
    ylabel('x_2');
    zlabel('f(x_1, x_2)');
    title('Exponential Function');
   
    figure(2)
    
    subplot(2,2,1)
    [C, h] = contour(X1, X2, y, [1 2 3 4 5 6 7 8 9 10]);
    set(h,'ShowText','on','TextStep',get(h,'LevelStep'))
    colormap hot
    axis equal
    xlabel('x_1');
    ylabel('x_2');
    title('Contour Plot');
    
    % perform gradient descent with different norms
    x0 = [-1; -1];
    
    % L1 Norm
    subplot(2,2,2)
    [C, h] = contour(X1, X2, y, [1 2 3 4 5 6 7 8 9 10]);
    colormap gray
    axis equal
    xlabel('x_1');
    ylabel('x_2');
   
    [x_series, fevalL1] = steepestDescent(x0, 1, 1E-4);

    plotSeries(x_series)
    hold on
    plot(x_series(1,length(x_series)), x_series(2,length(x_series)), 'r+', 'MarkerSize', 15);
    hold off
    title('Steepest Descent with L1 Norm');

    
    % L2 Norm
    subplot(2,2,3)
    [C, h] = contour(X1, X2, y, [1 2 3 4 5 6 7 8 9 10]);
    colormap gray
    axis equal
    xlabel('x_1');
    ylabel('x_2');
   
    [x_series, fevalL2] = steepestDescent(x0, 2, 1E-4);

    plotSeries(x_series)
    hold on
    plot(x_series(1,length(x_series)), x_series(2,length(x_series)), 'r+', 'MarkerSize', 15);
    hold off
    title('Steepest Descent with L2 Norm');

    
    % LP Norm
    subplot(2,2,4)
    [C, h] = contour(X1, X2, y, [1 2 3 4 5 6 7 8 9 10]);
    colormap gray
    axis equal
    xlabel('x_1');
    ylabel('x_2');
   
    [x_series, fevalLP] = steepestDescent(x0, 3, 1E-4);

    plotSeries(x_series)
    hold on
    plot(x_series(1,length(x_series)), x_series(2,length(x_series)), 'r+', 'MarkerSize', 15);
    hold off
    title('Steepest Descent with LP Norm');
    
    fevalL1
    fevalL2
    fevalLP
end

function plotSeries(x_series)
    hold on
    for i=1:length(x_series)
        plot(x_series(1,i), x_series(2,i), 'ko');
        if (i < length(x_series) - 1)
            dx = x_series(:,i+1) - x_series(:,i);
            quiver(x_series(1,i), x_series(2,i), dx(1), dx(2),0);
        end
    end
    hold off
end

function val = f(x)
    val = exp(x(1,:) + 3.*x(2,:) - 0.1) + exp(x(1,:) - 3.*x(2,:) - 0.1) + exp(-x(1,:) - 0.1);
end

function v = gradf(x)
    v = zeros(2,1);
    v(1) = exp(x(1,:) + 3.*x(2,:) - 0.1) + exp(x(1,:) - 3.*x(2,:) - 0.1) - exp(-x(1,:) - 0.1);
    v(2) = 3.*exp(x(1,:) + 3.*x(2,:) - 0.1) - 3.*exp(x(1,:) - 3.*x(2,:) - 0.1);
end

function H = hessianf(x)
    H = zeros(2,2);
    H(1,1) = exp(x(1,:) + 3.*x(2,:) - 0.1) + exp(x(1,:) - 3.*x(2,:) - 0.1) + exp(-x(1,:) - 0.1);
    H(1,2) = 3.*exp(x(1,:) + 3.*x(2,:) - 0.1) - 3.*exp(x(1,:) - 3.*x(2,:) - 0.1);
    H(2,1) = H(1,2);
    H(2,2) = 9.*exp(x(1,:) + 3.*x(2,:) - 0.1) + 9.*exp(x(1,:) - 3.*x(2,:) - 0.1);
end

function dx = L1Descent(gF)
    [tmp, idx] = max(abs(gF));
    ei = zeros(size(gF, 1), size(gF, 2));
    ei(idx) = 1;
    dx = -sign(gF(idx)) .* ei;
end

function dx = L2Descent(gF)
    dx = -gF;
end

function dx = LPDescent(gF, P)
    dx = -inv(P)*gF;
end

function [t, feval] = backtrackingLineSearch(x, dx, alpha, beta)
    t = 1;
    feval = 2;
    while f(x + t.*dx) > (f(x) + alpha .* t .* gradf(x)' * dx)
        t = t .* beta;
        feval = feval + 2;
    end
end

function [x_series, feval] = steepestDescent(x0, n, eps)

% initialization
    x_series = x0;
    xk = x0;
    xkk = x0;
    feval = 0;
    
% main optimization loop
    converged = false;
    while ~converged
        
        % get gradient of the function
        gF = gradf(xk);
        
        % compute steepest descent
        
        if n == 1
            dx = L1Descent(gF);
        elseif n == 2
            dx = L2Descent(gF);
        else
            dx = LPDescent(gF, hessianf(xk));
            % to be fair, we weight the evaluation of the hessian as 3
            % function evaluations
            feval = feval + 3;
        end
        
        % backtracking line search
        [t, fe] = backtrackingLineSearch(xk, dx, 0.4, 0.6);
        feval = feval + fe;
        
        xkk = xk + t .* dx;
        x_series = [x_series, xkk];
        if norm(xkk - xk) < eps
            converged = true;
        else
            xk = xkk;
        end
    end
end





