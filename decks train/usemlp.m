% Using mlp
function [out, outnet] = usemlp(inputs, net)
    outnet{1} = (inputs.*net{1});
    
    for i = 2:length(net)
        outnet{i} = logsig([outnet{i-1} -1]*net{i});
    end
	
    out = outnet{length(outnet)};
end