function net = createmlp(n,p,q)
    net{1} = ones(1,n);
    net{2} = rand(n+1,p(1));
    
    if(length(p)==1)
        i = 1;
    else
        for i = 2:length(p)
            net{i+1} = rand(p(i-1)+1,p(i));
        end
    end
    
    net{length(net)+1} = rand(p(i)+1,q);
end
