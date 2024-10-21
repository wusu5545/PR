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
	
	% L2 Norm
    subplot(2,2,3)
    [C, h] = contour(X1, X2, y, [1 2 3 4 5 6 7 8 9 10]);
    colormap gray
    axis equal
    xlabel('x_1');
    ylabel('x_2');
   
    %[x_series, fevalL2] = steepestDescent(x0, 2, 1E-4);

    %plotSeries(x_series)
    %hold on
    %plot(x_series(1,length(x_series)), x_series(2,length(x_series)), 'r+', 'MarkerSize', 15);
    %hold off
    %title('Steepest Descent with L2 Norm');

	% TODO: other norms, steepestDescent, ...
	
end


function val = f(x)
    val = exp(x(1,:) + 3.*x(2,:) - 0.1) + exp(x(1,:) - 3.*x(2,:) - 0.1) + exp(-x(1,:) - 0.1);
end

function v = gradf(x)

end

function H = hessianf(x)

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