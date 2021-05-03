function PreferenceLearningTest()
% Example on preference learning via Gaussian Processes
% Implementing algorithm in "Preference Learning with Gaussian Processes",
% by Chu, Ghahramani, ICML, 2005

        [fun,nvars,lb,ub] = def_fun();  % Define your underlying function (only used to generete synthetic data)  
        opts.M = 200;   % Number of pairwise comparisons for training
        opts.N = 200;    % Number of experiments for training
        Mtest = 100;    % Number of pairwise comparisons for testing
        Ntest = 100;     % Number of experiments for testing
        
        opts.hyper_opt = 1; % set opts.hyper_opt = 1 if kernel hyper-parameters should be selected by maximizing the marginal likelihood 

        % Initial values of kernel hyper-parameters
        opts.SE.l2 = 10*ones(nvars,1);   %length-scale square
        opts.SE.sigmaf2 = 10; %kernel variance
        opts.sigmae2 = 1; % noise variance

        % Set parameters tocompute f_MAP, then used for Laplace approximation
        opts.maxiter_fMAP = 1000;   % Maximum number of iterations for MAP optimization 
        opts.tol = 1e-3;            % Set tolerance on the norm of the gradient to terminate optimization algorithm 
        opts.opt_var = 1;           % set 1 for Newton-Raphsod algorithm. Otherwise, gradient method is used 
        opts.nsearch = 10;          % number of grid points for exact line-search 
        opts.f0 = zeros(opts.N,1);  % Initial condition for optimization (Nx1-vector)
        opts.alpha = 0.0001;        % Regularization parameter for Kernel matrix
        opts.nvars = nvars;
  
        
        % data generation
 
 
        %% Generate random training samples 
        for ind=1:opts.nvars
            x(:,ind)= rand(opts.N,1).*(ub(ind)-lb(ind))'+lb(ind);
        end
        D.X = x;

        index = ceil(rand(opts.M,2)*opts.N); % Randomly select M pairs of indexes between 1 and N
        index(1,:) = [1, 2]; % Just to avoid (later) possible duplications
        D.Xp = zeros(opts.M,2);
        for ind = 1:opts.M

            if fun(x(index(ind,1),:)) >= fun(x(index(ind,2),:))
                D.Xp(ind,1:2) = index(ind,1:2); % D.Xp(ind,1:2) means that x in the first column is preferred to x in the second column
            else
                D.Xp(ind,1:2) = [index(ind,2), index(ind,1)];
            end

            if D.Xp(ind,1) == D.Xp(ind,2)
                D.Xp(ind,1:2) = D.Xp(ind-1,1:2);
            end
        end
    
        %% Generate  Test data
        
       clear x, index
        for ind=1:opts.nvars
            x(:,ind)= rand(Ntest,1).*(ub(ind)-lb(ind))'+lb(ind);
        end
        Xtest = x;

        index = ceil(rand(Mtest,2)*Ntest); % Randomly select M pairs of indexes between 1 and N
        index(1,:) = [1, 2]; % Just to avoid (later) possible duplications
        Xptest = zeros(Mtest,2);
        for ind = 1:Mtest

            if fun(x(index(ind,1),:)) >= fun(x(index(ind,2),:))
                Xptest(ind,1:2) = index(ind,1:2); % D.Xp(ind,1:2) means that x in the first column is preferred to x in the second column
            else
                Xptest(ind,1:2) = [index(ind,2), index(ind,1)];
            end

            if Xptest(ind,1) == Xptest(ind,2)
                Xptest(ind,1:2) = Xptest(ind-1,1:2);
            end
        end
        
        
        %% Learning
        %D (below) should contain: D.X (experiments, nxN matrix, where n is the number of features); D.Xp (Mx2 matrix). First column preferred to second one 
        opts.D = D;
        
        [D, opts] = PreferenceLearning(opts);
        [m_vec, s_vec, s2] = GP_mu_s(Xtest,D,opts); %compute mean and standard deviation of the esimated GP on the test data
        
        %(m_vec(Xptest(ind,1))-m_vec(Xptest(ind,2)))/(2* )
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

