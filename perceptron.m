%% perceptron - neural network 
% [w, erro, y_est, errors] = perceptron(x, y, alpha, tries)
% - x: data for training
% - y: data out
% - alpha: learning rate
% - tries: tries
%
% - w: weighted estimated
% - erro: error estimated
% - y_est: 'y' estimated
% - errors: all errors in the last training
%
% Note this perceptron is not ideal because it is not using error for finish training process.
function [w, erro, y_est, errors] = perceptron(x, y, alpha, tries)
    [sizeY, sizeX] = size(x);
    w = [rand(1,sizeX) 1]';
    mat = [x ones(sizeY,1)];

    errors = [];
    cont = 1;
    erro(cont) = norm(y-mat*w);
    while(erro(cont)>0.001 && tries > 0)
        for i = 1:sizeY
            errors(i) = y(i) - mat(i,:)*w;
            if(abs(errors(i))>0.01) % nao usei degrau
                w = w + (alpha*errors(i)*mat(i,:))';
            end
        end
       
        tries = tries - 1;
        cont = cont + 1;
        erro(cont) = norm(errors);
    end
    y_est = mat*w;
end
