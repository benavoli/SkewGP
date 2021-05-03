function [l2, sigmaf2, sigmae2] = BO_hyp(D,opts)
% Optimize hyper-parameters via Bayesian optimization.
%
% The kernel is defined as sigmaf2*exp(-0.5||x_i-x_j||^2/l2) and the 
% optimized hyper-parameters are: sigmaf2, l2, and noise variance sigmae2
%
% Inputs: 
%       D: structure with inputs x and pairwise preference Xp
%       opts: structure with parameters used by the optimization algorithm
%
% (C) 2019 D. Piga, Lugano, July 5, 2019


 bfun = @(theta) BO_MAP(theta,D,opts);

%     opt_vars = [optimizableVariable('l2', [0.00001,20],'Type','real'), ...
%                     optimizableVariable('sigmaf2', [100,200],'Type','real'), ...
%                     optimizableVariable('sigmae2', [opts.sigmae2,opts.sigmae2+0.0001],'Type','real')];

    opt_vars = [];

    for indV=1:opts.nvars
        s = ['l2', num2str(indV)];
        opt_vars = [opt_vars, optimizableVariable(s, [0.00091,exp(5)^2],'Type','real')]    
    end
    
    

    opt_vars = [opt_vars, ...
                optimizableVariable('sigmaf2', [0.00091,exp(4.1)],'Type','real'), ...
                optimizableVariable('sigmae2', [0.5,0.501],'Type','real')];


    t0=tic;
    results = bayesopt(bfun,opt_vars,...
        'Verbose',1,...
        'AcquisitionFunctionName','lower-confidence-bound',... 'expected-improvement' %-plus',...
        'IsObjectiveDeterministic', true,... % simulations with noise --> objective function is not deterministic
        'MaxObjectiveEvaluations', 100,...#*opts.nvars,...
        'MaxTime', inf,...
        'NumCoupledConstraints',0, ...
        'NumSeedPoint',40,...
        'GPActiveSetSize', 300,...
        'PlotFcn',{}); %@plotMinObjective});%,@plotObjectiveEvaluationTime}); %);
    t2=toc(t0);

    xopt3 = results.XAtMinObjective;
    fopt3=results.MinObjective;
        
    
    l2 = [];
    for indV = 1:opts.nvars
        %s = ['l2', num2str(indV)];
        sl2 = eval(['xopt3.l2', num2str(indV)]);
        l2 = [l2,sl2];
    end
    
    
    
    %l2=xopt3.l2;
    sigmaf2=xopt3.sigmaf2;
    sigmae2=xopt3.sigmae2;
    
