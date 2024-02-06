%--------------------------------------------------------------------------
% This function takes a DxN matrix of N data points in a D-dimensional 
% space and returns a NxN coefficient matrix of the low-rank + sparse representation 
% of each data point in terms of the rest of the points
% Y: DxN data matrix
% thr1: stopping threshold for the coefficient error ||J-C||
% thr2: stopping threshold for the linear system error ||X-XJ||
% maxIter: maximum number of iterations of ADMM
% C1: NxN low-rank+sparse coefficient matrix
%--------------------------------------------------------------------------
% Copyright @ Ivica Kopriva, 2016
%--------------------------------------------------------------------------

function C1 = admm_elrassc_func(X,alpha,tau,lam_elra,thr,maxIter)

if (nargin < 2)
    % default subspaces are linear
    affine = false; 
end
if (nargin < 3)
    % default regularizarion parameters
    alpha = 800;
end
if (nargin < 4)
    % default coefficient error threshold to stop ADMM
    % default linear system error threshold to stop ADMM
    thr = 2*10^-4; 
end
if (nargin < 5)
    % default maximum number of iterations of ADMM
    maxIter = 200; 
end

if (length(alpha) == 1)
    alpha1 = alpha(1);
    alpha2 = alpha(1);
    alpha3 = alpha(1);
elseif (length(alpha) == 3)
    alpha1 = alpha(1);
    alpha2 = alpha(2);
    alpha3 = alpha(3);
end

if (length(thr) == 1)
    thr1 = thr(1);
    thr2 = thr(1);
    thr3 = thr(1)
elseif (length(thr) == 3)
    thr1 = thr(1);
    thr2 = thr(2);
    thr3 = thr(3);
end

N = size(X,2);
D = size(X,1);

% setting penalty parameters for the ADMM
mu1 = alpha1 * 1; % /computeLambda_mat(X);
mu2 = alpha2 * 1;
mu3 = alpha3 * 1;

% initialization
A = inv(mu1*(X'*X)+(mu2+mu3)*eye(N));
C1 = zeros(N,N);
C2 = zeros(N,N);
Lambda1 = zeros(D,N);
Lambda2 = zeros(N,N);
Lambda3 = zeros(N,N);

a = 0.6/lam_elra;

%matrix size
[m,n]=size(X);
k = min(m,n)-1;
err1 = 10*thr1; err2 = 10*thr2; err3 = 10*thr3;
i = 1;

% ADMM iterations
 while ( err1(i) > thr1 && i < maxIter )
        % updating J
        J = A * (mu1*(X'*X) + mu2*C2 + mu3*C1 + X'*Lambda1 - Lambda2 - Lambda3);
        J = J - diag(diag(J));   %% ???
        % updating C2 by soft thresholding
        C2=wthresh(J+Lambda2/mu2,'s',tau/mu2);
        C2 = C2 - diag(diag(C2));
        
        % updating C1 by enhanced low-rank approximation
        [U,Sig,V]=svdsecon(J+Lambda3/mu3,k);
    
       % Apply enhanced low-rank approximation
        
        sig = diag(Sig)';
        tmp = (sig-lam_elra/mu3)/(1-a*lam_elra/mu3);  % lam_era/mu3 ????
        sigm = max([tmp; zeros(1,length(sig))]);
        tmp = [sig; sigm];
        Sig = diag(min(tmp).*sign(sig));
        C1 = U*Sig*V'; 
    
        % updating Lagrange multipliers
        Lambda1 = Lambda1 + mu1 * (X - X*J);
        Lambda2 = Lambda2 + mu2 * (J - C2);
        Lambda3 = Lambda3 + mu3 * (J - C1);
        % computing errors
        err1(i+1) = errorCoef(J,C1);
        err2(i+1) = errorLinSys(X,J);
        %
        i = i + 1;
 end
 fprintf('err1: %2.4f, err2: %2.4f, iter: %3.0f \n',err1(end),err2(end),i);
 
