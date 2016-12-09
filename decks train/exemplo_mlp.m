% find(strcmp(upper(types),typesKeys{1}))
% clear all
close all
clc
load DadosExemplo

%% Exemplo
data=x;
% data(:,6)=ones(size(data,1),1);
alpha=0.05;
hl = [10];
tries = 1000;
rt = y';
ol = size(rt, 2); % number of output layers
net = createmlp(length(data(1,:)),hl,ol);
erros = [];
momentum = 1;
for cont = 1:tries
	ids = randperm(size(data,1));
	while length(ids)>0
		id = ids(1); ids(1) = [];
		
		[out, outnet] = usemlp( data(id,:), net);
		% output layer erro
		e = rt(id,:) - out;
		% delta output layer
		delta = {};
		delta{length(outnet)} = out.*(1-out).*(e);
		for i = length(outnet)-1:-1:1
			delta{i} = (outnet{i}.*(1-outnet{i})).*(delta{i+1}*net{i+1}(1:end-1,:)');
%           delta{i} = [(outnet{i}.*(1-outnet{i})) 1].*[delta{i+1} 1]*net{i+1};
%             delta{i} = [(outnet{i}.*(1-outnet{i})) 1].*(delta{i+1}*net{i+1}');
		end
		
		% adjust weights
        net{end} = net{end} + alpha*[outnet{end-1} -1]'*delta{end};
		for i = length(net)-1:-1:2
			w = net{i};
			w = w*momentum + alpha*[outnet{i-1} -1]'*delta{i};
			net{i} = w;
        end
		erro(id,:) = rt(id,:) - usemlp( data(id,:), net);
	end
	erros(cont) = mean(diag(erro'*erro))/size(erro,1);
end

% verifica erros
oute = [];
outend = {};
for k = 1:size(data, 1)
	[oute(k,:),outend{k}] = usemlp( data(k,:), net);
end
erro = rt - oute;
oute
plot(erros)