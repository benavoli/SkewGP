function  [mu_x, s_x, s2_x] = GP_mu_s(x,D,opts)
% Compute mean value standard deviation of the surrogate GP
% 
% Inputs:
%     x: inputmatrix
%     D: training data
%     opts: structure with parameters used by the optimization algorithm
% 
% Outputs:
%     mu: mean
%     s_x: standard deviation 
%     
% (C) 2019 D. Piga, Lugano, July 5, 2019

        
        Kt =  build_Kernel(D.X,x,opts); 
        mu_x = Kt'*(opts.Sigma\D.fMAP);
        
        Sigmat = build_Kernel(x,x,opts);
        s2_x = Sigmat - Kt'*( (opts.Sigma+inv(opts.LambdaMAP+0.000001*eye(opts.N))) \Kt);      
        s_x = sqrt(diag(s2_x));
end