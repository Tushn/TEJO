%% Mlp function
% [net, erro, y_est, erros] = mlp(data, hl, ol, tries)
% Inputs:
% 	- data: data of inputs
% 	- rt: result training
% 	- hl: number of hidden layers
%	- tries: number max of tries
% 	- alpha: learning rate
% function [net, erro, y_est, erros] = mlp(data, rt, alpha, hl, tries)
% function [net, outnet, erro, erros] = mlp(data, rt, alpha, hl, tries)
function [net, outnet, erros] = mlp(data, rt, alpha, hl, tries)
	% normalize
	% for i = 1:size(data,2)
		% data(:,i) = data(:,i)/norm(data(:,i));
	% end
	% rt = rt/norm(rt);
	
	ol = size(rt, 2); % number of output layers
	net = createmlp_new(length(data(1,:)),hl,ol);
	erros = [];
	momentum = 1;
	for cont = 1:tries
		ids = randperm(size(data,1));
		cont
		while length(ids)>0
			id = ids(1); ids(1) = [];
			
			[out, outnet] = usemlp_new( data(id,:), net);
			% Erros calculated
			% output layer erro
			e = rt(id,:) - out;
			% delta output layer (errors)
			delta = {};
			delta{length(outnet)} = out.*(1-out).*(e);
			for i = length(outnet)-1:-1:1
				delta{i} = (outnet{i}.*(1-outnet{i})).*(delta{i+1}*net{i+1}(1:end-1,:)');
			end
			
			% Weights calculated
			% adjust weights
			net{end} = net{end} + alpha*[outnet{end-1} -1]'*delta{end};
			for i = length(net)-1:-1:2
				w = net{i};
				w = w*momentum + alpha*[outnet{i-1} -1]'*delta{i};
				net{i} = w;
			end
			erro(id,:) = rt(id,:) - usemlp_new( data(id,:), net);
		end
		erros(cont) = mean(diag(erro'*erro))/size(erro,1);
	end

	% Verify errors for all data
	oute = [];
	outend = {};
	for k = 1:size(data, 1)
		[oute(k,:),outend{k}] = usemlp_new( data(k,:), net);
	end
	oute
end
