function log_pD = BO_MAP(theta,D,opts)
% Compute log of the evidence p(D|theta). Used to optimize the
% hyper-parameters via Bayesian optimization
%
% The kernel is defined as sigmaf2*exp(-0.5||x_i-x_j||^2/l2) and the 
% optimized hyper-parameters are: sigmaf2, l2, and noise variance sigmae2
%
% Inputs: 
%       theta: structure with hyper-parameters sigmaf2, l2, sigmae2
%       D: structure with inputs x and pairwise preference Xp
%       opts: structure with parameters used by the optimization algorithm
%
% (C) 2019 D. Piga, Lugano, July 5, 2019

    opts.SE.l2 = [];
    for indV = 1:opts.nvars
        %s = ['l2', num2str(indV)];
        sl2 = eval(['theta.l2', num2str(indV)]);
        opts.SE.l2 = [opts.SE.l2,sl2];
    end
    
    opts.SE.sigmaf2 = theta.sigmaf2;
    opts.sigmae2 = theta.sigmae2;

    % Compute Kernel Matrix
    K = build_Kernel(D.X,D.X,opts);
    N = size(D.X,1);
    opts.Sigma = K+opts.alpha*eye(N); % Kernel + regularization
    opts.Sigmainv = inv(opts.Sigma);
    [fMAP, gMAP, HMAP, betaMAP, LambdaMAP, L, log_pD] = compute_fMAP(D,opts);
    log_pD = -log_pD;

end