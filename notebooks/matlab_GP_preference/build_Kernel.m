function K = build_Kernel(X1,X2,opts)

n1 = size(X1,1);
n2 = size(X2,1);

K = ones(n1,n2);

    for ind1 = 1:n1
        
        x1 = X1(ind1,:)';
        
        for ind2 = 1:n2
            x2 = X2(ind2,:)';
            
            K(ind1,ind2) = SE_ij(x1,x2,opts); 
            
        end
    end

end


function Kij = SE_ij(x1,x2,opts)

    Kij = opts.SE.sigmaf2*exp(-0.5*(x1-x2)'*inv(diag(opts.SE.l2))*(x1-x2));

end
