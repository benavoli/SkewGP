function [fMAP, gMAP, HMAP, betaMAP, LambdaMAP, L, log_pD] = compute_fMAP(D,opts)
% Compute MAPof the latent preference function f
%
% Inputs: 
%       D: structure with inputs x and pairwise preference Xp
%       opts: structure with parameters used by the optimization algorithm
%
% Outputs: 
%       fMAP: MAP value of the latent function f
%       gMAP: Gradient at f=fMAP
%       betaMAP: beta parameter (see paper Chu, Ghahramani, ICML, 2005, eq. 11)
%       LambdaMAP: Lambda matrix at f = fMAP (see paper Chu, Ghahramani, ICML, 2005)
%       L: -log of the posterior p(f|D) (up to constant 1/p(D)) 
%       log_PD: approximation of the log of the evidence p(D) (used for hyper-parameter selection)
%
% (C) 2019 D. Piga, Lugano, July 5, 2019

    [N,d] = size(D.X);
    
    %% Compute fMAP
    % Initial value of f
    f = opts.f0;

    for i = 1:opts.maxiter_fMAP
        [grad_f, H_f, flag_c] = compute_grad_H_loss(D,f,opts);      % Compute Gradient and Hessian at f. If flag_c==1, then the cumulative is equal to zero and the Hesssian is not well approximated. 
                                                                     %In this case, at that iteration, gradient method will be used instead of Newton-Raphson 
        
        L = compute_log_posterior(D,f,opts);                        % Compute -log of the posterior p(f|D) (up to the constant term 1/p(D))
        
        fprintf('Iteration: %d, Cost: %2.4f, Gradient: %2.6f \n',i,L,norm(grad_f))
        
        if opts.opt_var == 1 && flag_c == 0 % Use Newton method
            t = linsearch(D,f,H_f,grad_f,opts,flag_c);
            deltaf =  - t*(H_f\grad_f);
        else                                % Use gradient descent
            t = linsearch(D,f,H_f,grad_f,opts,flag_c);
            deltaf = -t*grad_f;
        end
        f = f + deltaf;  % Update f
        
        if norm(grad_f)<=opts.tol
           break 
        end
        
        if abs(L)>=1e20
            L = 1e21;
           break 
        end
        
    end % for i = 1:opts.maxiter_fMAP
    %%
    
    % Extract Gradient and Hessian at the optimum
    [gMAP, HMAP, flag_c,  betaMAP, LambdaMAP] = compute_grad_H_loss(D,f,opts); 
    fMAP = f;
    
    if abs(L)<1e20
        L = compute_log_posterior(D,fMAP,opts);
    end
    
    log_pD =  -L -0.5*log(det(eye(N)+opts.Sigma*LambdaMAP)); %(approximation of) the log of the evidence p(D) (used for hyper-parameter tuning)

   
end


function [grad_f, H_f, flag_c, beta, Lambda_MAP] = compute_grad_H_loss(D,f,opts)
% Compute gradient and Hessian of the Loss w.r.t. the latent function f, at
% a given point f

flag_c = 0;
M = size(D.Xp,1);
N = size(D.X,1);

grad_f = zeros(N,1);
H_f = zeros(N,N);

for ind = 1:M
   
    v = D.Xp(ind,1); 
    u = D.Xp(ind,2);
    
    den2 = 2*opts.sigmae2;
    den = sqrt(den2);
    
    z = (f(v) - f(u))/den;
    if z <-25
        flag_c = 1;
        z = -25;
    end
        
    normal = normpdf(z);
    cumulative = normcdf(z);
    
    if cumulative == 0
         flag_c = 1;
         grad1 = abs(f(v))*0.1;
         grad2 = abs(f(u))*0.1;
         grad_f(v) = grad_f(v) - grad1;
         grad_f(u) = grad_f(u) + grad2;

    else
         grad_f(v) = grad_f(v) - 1/den*normal/cumulative;
         grad_f(u) = grad_f(u) + 1/den*normal/cumulative;
    end
    

  
 
    H_f(v,u) = H_f(v,u)-1/den2*(normal^2/cumulative^2 + z*normal/cumulative);
    H_f(u,v) = H_f(v,u);
    H_f(v,v) = H_f(v,v)+1/den2*(normal^2/cumulative^2 + z*normal/cumulative);
    H_f(u,u) = H_f(u,u)+1/den2*(normal^2/cumulative^2 + z*normal/cumulative);

    
    
end

    beta = -grad_f; %-(grad_f - (opts.Sigma\f));
    Lambda_MAP = H_f; %-(H_f - opts.Sigmainv);
    
    grad_f = grad_f + 1*(opts.Sigma\f);%zeros(N,1);
    H_f = H_f + opts.Sigmainv; %zeros(N,N);

end



function L = compute_log_posterior(D,f,opts)

    M = size(D.Xp,1);
    L = 0.5*f'*opts.Sigmainv*f;

    
    for ind = 1:M

        v = D.Xp(ind,1); 
        u = D.Xp(ind,2);

        den2 = 2*opts.sigmae2;
        den = sqrt(den2);

        z = (f(v) - f(u))/den;
        normal = normpdf(z);
        cumulative = normcdf(z);
        if cumulative==0
            cumulative = 1e-55;
        end

        L = L - log(cumulative);
    end

end


function t = linsearch(D,f,H_f,grad_f,opts, flag_c)

    if opts.opt_var==0 || flag_c == 1
        tvec = linspace(0,1,opts.nsearch);
        tvec(1) = 0.01;
    else
        tvec = linspace(0.2,1,opts.nsearch);
    end
        
    for ind = 1:1 %opts.nsearch
        %t = tvec(ind);
        t = 1;
        if opts.opt_var == 1 && flag_c==0
            deltaf =  - t*(H_f\grad_f);
        else
            deltaf = -t*grad_f;
        end
        fn = f + deltaf; 
    
    L(ind) = compute_log_posterior(D,fn,opts);
    end
    
    [minval, indmin] = min(L);
    
    t = tvec(indmin);   
end
