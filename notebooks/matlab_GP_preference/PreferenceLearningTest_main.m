function PreferenceLearningTest()
% Example on preference learning via Gaussian Processes
% Implementing algorithm in "Preference Learning with Gaussian Processes",
% by Chu, Ghahramani, ICML, 2005

        %[fun,nvars,lb,ub] = def_fun();  % Define your underlying function (only used to generete synthetic data)  
        X_train = table2array(readtable('X_train1.csv','ReadVariableNames', false));
        y_train = table2array(readtable('y_train1.csv','ReadVariableNames', false));
        pref_train = table2array(readtable('pref_train1.csv','ReadVariableNames', false));
        pref_train = pref_train + 1;
        opts.M = size(pref_train,1);   % Number of pairwise comparisons for training
        opts.N = size(y_train,1);    % Number of experiments for training
        
        
        X_test = table2array(readtable('X_test1.csv','ReadVariableNames', false));
        y_test = table2array(readtable('y_test1.csv','ReadVariableNames', false));
        pref_test = table2array(readtable('pref_test1.csv','ReadVariableNames', false));
        pref_test = pref_test + 1;
        Mtest = size(pref_test,1);    % Number of pairwise comparisons for testing
        Ntest = size(y_test,1);     % Number of experiments for testing
        
        opts.hyper_opt = 1; % set opts.hyper_opt = 1 if kernel hyper-parameters should be selected by maximizing the marginal likelihood 

        % Initial values of kernel hyper-parameters
        opts.SE.l2 = 0.5^2;   %length-scale square
        opts.SE.sigmaf2 = 2; %kernel variance
        opts.sigmae2 = 1; % noise variance

        % Set parameters tocompute f_MAP, then used for Laplace approximation
        opts.maxiter_fMAP = 1000;   % Maximum number of iterations for MAP optimization 
        opts.tol = 1e-3;            % Set tolerance on the norm of the gradient to terminate optimization algorithm 
        opts.opt_var = 1;           % set 1 for Newton-Raphsod algorithm. Otherwise, gradient method is used 
        opts.nsearch = 10;          % number of grid points for exact line-search 
        opts.f0 = zeros(opts.N,1);  % Initial condition for optimization (Nx1-vector)
        opts.alpha = 0.0001;        % Regularization parameter for Kernel matrix
        opts.nvars = 1;
  
        
        % data generation
 
 
        %% Generate random training samples 
        
        D.X = X_train;
        D.Xp = pref_train;

    
        %% Generate  Test data
        
        Xtest = X_test;
        Xptest = pref_test;        
        
        %% Learning
        %D (below) should contain: D.X (experiments, nxN matrix, where n is the number of features); D.Xp (Mx2 matrix). First column preferred to second one 
        opts.D = D;
        
        [D, opts] = PreferenceLearning(opts);
        [m_vec, s_vec] = GP_mu_s(Xtest,D,opts); %compute mean and standard deviation of the esimated GP on the test data
        
        
        %Assess performce on the test dataset
        for ind=1:size(Xptest,1)
            
             
            if m_vec(Xptest(ind,1)) >= m_vec(Xptest(ind,2))
                ris(ind) = 1; % D.Xp(ind,1:2) means that x in the first column is preferred to x in the second column
            else
                ris(ind) = 0;
            end
            
            
        end
        
        fprintf('Achieved accuracy: %2.2f', mean(ris));
 
end

%%



function [D, opts] = PreferenceLearning(opts)
% Preference learning via Gaussian Processes
%
%
% Inputs: 
%       opts:  structure with parameters used by the optimization algorithm
%
% Outputs: 
%       xopt: best value of the input x
%       fopt: mean value of the latent preference function at x = xopt
%       out:  structure with tested points and observed preferences
%
% (C) 2019 D. Piga, Lugano, July 5, 2019



%% 
% If initial samples are not provided, generate here sythetic data
    D = opts.D;
    out.X = D.X;
    out.Xp = D.Xp;
    N = opts.N;
    M = opts.M;

        % Optimize hyper-parameters through Bayesian optimization
        
    [opts.SE.l2, opts.SE.sigmaf2, opts.sigmae2] = BO_hyp(D,opts);
     
        
        
        
        K = build_Kernel(D.X,D.X,opts);   
        opts.Sigma = K+opts.alpha*eye(opts.N); % Kernel
        opts.Sigmainv = inv(opts.Sigma);
        [fMAP, gMAP, HMAP, betaMAP, LambdaMAP, L, log_pD] = compute_fMAP(D,opts);
        opts.LambdaMAP = LambdaMAP;
        

        D.fMAP = fMAP; 
         % Fit GP of the posterior p(f|D). Returns mean and standard deviation
         
         
 
         
    
end

function [f,nvars,lb,ub] = def_fun()

        nvars = 2;            % Number of variables
        lb=-6*ones(nvars,1);  % Lower bounds on the optimization variables
        ub=6*ones(nvars,1);   % Upper bounds on the optimization variables
        f=@(x) (x(:,1).^2+x(:,2)-11).^2+(x(:,1)+x(:,2).^2-7).^2;

end



