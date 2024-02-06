%--------------------------------------------------------------------------
% This function takes a DxN matrix Y of N data points in a D-dimensional 
% space and:
% 1. approximates it with a DxN low-rank and sparse matrix X
% 2. based on approximation X it returns NxN graph adjacency matrix C to be
%    used for spectral clustering
%
% lam_0: low-rank approximation related regularization factor
% lam_1: sparsity related regularization factor
% c: in (0, 1) interval to regulate a0 and a1 thresholding constants
% thr: stopping criterion
% maxIter: maximal number of interation
% kernel: string for the similarity measure for graph adjacency matrix
% var1, var2: kernel parameters
%
% W: NxN graph adjacency matrix
%--------------------------------------------------------------------------
% Copyright @ Ivica Kopriva, 2016
%--------------------------------------------------------------------------

function W = admm_elrsa_sc_func(Y,lam_0, lam_1,c, mu,thr,maxIter, kernel, var1, var2)

if (nargin < 2)
    lam_0 = 1;
end
if (nargin < 3)
    lam_1 = 1;
end
if (nargin < 4)
    mu = 1.5;
end
if (nargin < 5)
    % default coefficient error threshold to stop ADMM
    % default linear system error threshold to stop ADMM
    thr = 2*10^-4; 
end
if (nargin < 6)
    % default maximum number of iterations of ADMM
    maxIter = 200; 
end

if mu < 1
    display('Lagrange multiplier has to be greater than 1')
    mu = 1.5;
end

N = size(Y,2);
L = size(Y,1);
k = min(L,N)-1;

% setting thresholding constants
a0 = c/lam_0;
a1 = (1-a0*lam_0)/lam_1;

% initialization
D = zeros(L,N);
Z = D;
err1 = 10*thr;
i = 1;

% ADMM iterations
 while ( err1(i) > thr && i < maxIter )
        
        % updating Y_th
        Y_th = (Y + mu*(Z + D))/(1+mu);
        
        % updating X - apply enhanced sparse approximation       
        tmp = (abs(Y_th)-lam_1/(1+mu))/(1-a1*lam_1/(1+mu));  
        tmp = max(0,tmp);
        X = min(abs(Y_th),tmp).*sign(Y_th);      
        
        % updating Z - apply enhanced low-rank approximation
        [U,Sig,V]=svdsecon(X-D,k);
        sig = diag(Sig)';
        tmp = (sig-lam_0/mu)/(1-a0*lam_0/mu);  
        sigm = max(tmp,0);
        Sig = diag(min(sig,sigm).*sign(sig));
        Z = U*Sig*V'; 
    
        % Updating D
        D = D - (X - Z);
        
       % computing errors
        err1(i+1) = errorCoef(X,Z);
        %
        i = i + 1;
 end
 fprintf('err1: %2.4f, iter: %3.0f \n',err1(end),i);
 
 xn=norm(Y-X)/L/N
 
 % INITIALIZATION

if(var1<=0)
    error('Parameter var1 must be positive.');
end

% parameters of kernel
switch(kernel)
    case 'linear'
        W = X'*X;
    case 'polynomial'
        dp = var1;       % degree
        bp = var2;       % offset
        if(bp<0)
            error('Parameter c (offset) must be nonnegative.');
        end
        W = power(X'*X + bp,dp); 
    
    case 'gaussian'
        sig2 = var1;   
        g = waitbar(0,'Subspace projection ...');
        for n=1:N
              P = X-repmat(X(:,n),1,N);   
              P = sum(P.*P);
              W(n,:)=exp(-P/2/sig2);
              waitbar(n/N,g)
        end
        close(g)
         
    case 'exponential'
        sig2 = var1;
        W = exp(X'*X/2/sig2);    

    case 'sigmoid'
        sig = var1;
        c = var2;
        W = tanh(X'*X/sig + c);
                
    case 'laplace'
        sig2 = var1;
        g = waitbar(0,'Subspace projection ...');
        for n=1:N
            P = X-repmat(X(:,n),1,N);   
            P = sqrt(sum(P.*P));
            W(n,:)=exp(-P/2/sig2);
            waitbar(n/N,g)
        end
        close(g)
        
    case 'multiquad'
        c = var1;
        
    case 'invmultiquad'
        c = var1;
        
    otherwise
        error('Unknown kernel')
end
