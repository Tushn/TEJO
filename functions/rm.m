%% Linear regression using at least square (ALS)
% [b, y_est, erro] = rm(x,y)
% - x: data for training
% - y: data for out
% 
% - b: constants for equations
% - y_est: 'y' estimated
% - erro: error is norm(y-y_est)
function [b, y_est, erro] = rm(x, y)
    x(:,size(x,2)+1) = ones(size(y,1),1);
    b = pinv(x'*x)*x'*y;
    y_est = x*b;
    erro = norm(y-y_est);
end
