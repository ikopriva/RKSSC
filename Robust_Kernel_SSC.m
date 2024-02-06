%
% Robust version of nonlinear kernel SSC algorithm
% (c) Ivica Kopriva, January, 2024.
%
% Please cite the follwowing paper: I. Kopriva, "", arXiv

clear all; close all

addpath data
addpath measures
addpath SSC_ADMM_v1.1

dataset ='MNIST'; % 'EYaleb' or 'MNIST'

% Parameters of the inear SSC algorithm in RKHS 
outlier = false;  %noise/error model. true: L1-norm of the error term; 
                 %                   false: L2-norm of the error term.
r = 0; affine = false; rho = 1;                 

kernel = 'gauss'; % 'gauss' - Gaussian kernel; 'poly' - Polynomial kernel                 

if dataset == "EYaleb"
    load YaleBCrop025.mat

    p = 2016; % samples dimension
    n = 64; % number of samples per group 
    n_in = 45;  % number of samples for in_sample part
    n_out = 19; % number of samples for out_of_sample part
    nc = 38; % dataset parameters
    dsubspace=9; % subspace dimension

    X = reshape(Y,[p,n*nc]);
    clear Y I 
    
    % parameters of (robust) kernel SSC algorithm
    if outlier == true
        switch(kernel)
            case 'gauss'
                RMAX=1600; R=1510; sig2=810;
                alpha = sqrt(R/RMAX)*4;
            case  'poly'
                RMAX=1600; R=600; b=1; d=1;
                alpha = sqrt(R/RMAX)*6;
        end
    elseif outlier == false
        switch(kernel)
            case 'gauss'
                R=350; sig2=4000; alpha=6;
            case 'poly'
                R=1524; b=4; d=1; alpha=7;
        end
    end
elseif dataset == "MNIST"
    images = loadMNISTImages('t10k-images.idx3-ubyte');
    labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
    [labels,IX] = sort(labels');
    X = images(:,IX);   % Sort images according to subspaces/digits
    clear images

    p = 784;   % sample dimension
    n = 1000;  % number of samples (digits) per group
    n_in = 200;  % number of samples for in_sample part
    n_out = 200; % number of samples for out_of_sample part
    nc = 10; % number of clusters (digits)
    dsubspace=12; % subspace dimension

    % parameters of (robust) kernel SSC algorithm
    if outlier == true
        switch(kernel)
            case 'gauss'
                RMAX=2000; R=380; sig2=0.9;
                alpha = sqrt(R/RMAX)*8;
            case 'poly'
                RMAX=2000; R=600; b=0; d=11;
                alpha = sqrt(R/RMAX)*112;
        end
    elseif outlier == false
        switch(kernel)
            case 'gauss'
                R=650; sig2=1.3; alpha=4;
            case 'poly'
                R=100; b=1; d=5; alpha=3;
        end
    end
end

%% Repeat 100 times for best selection

n_trial = 10;
for i_trial = 1:n_trial
    
    fprintf('Iter %d\n',i_trial);

    tstart = tic;
    
    % generate a problem instance
    labels_in = [];    
    labels_out = [];
    X_in = [];   
    X_out = [];
    
    switch(dataset)
        case 'EYaleb'
            for l=1:nc
                indexes=randperm(n);
                ind_in=indexes(1:n_in);    % n_in images per digit for in-sample data
                ind_out=indexes(n_in+1:n_in+n_out);  % n_out images per digit for in-sample data
                labels_in = [labels_in, ones(1,n_in)*l];
                X_in = [X_in, X(:,(l-1)*n + ind_in)]; % In_sample dataset
                labels_out = [labels_out, ones(1,n_out)*l];
                X_out = [X_out, X(:,(l-1)*n + ind_out)]; % Out_of_sample dataset
            end
        case 'MNIST'
            for l=1:nc
                ind = randperm(n);
                ind_in=ind(1:n_in);    % n_in images per digit for in-sample data
                ind_out=ind(n_in+1:n_in+n_out);  % n_out images per digit for in-sample data
                labels_in = [labels_in, ones(1,n_in)*l];
                X_in = [X_in, X(:,(l-1)*n + ind_in)]; % In_sample dataset
                labels_out = [labels_out, ones(1,n_out)*l];
                X_out = [X_out, X(:,(l-1)*n + ind_out)]; % Out_of_sample dataset
            end
    end

    load X_control

    N_in = size(X_in,2);
    I_ONES = (eye(N_in)-ones(N_in,N_in)/N_in);
    
    % going to RKHS
    Xtrain=normc(X_in)';

   switch(kernel)
       case 'gauss' % Gaussian kernel
           D = pdist2(Xtrain,Xtrain);
           K=exp(-(D.*D)/2/sig2);
           clear D
       case 'poly' % poly kernel
           K=(Xtrain*Xtrain' + b).^d;
       otherwise
           error('Unknown parameter "ktype".')
   end

    K_uncentered=K;
    K = I_ONES*K*I_ONES;  % centering
    K_ones = K_uncentered*ones(N_in,1)/N_in;

    [U,LAM]=eig(K);

%     dLAM = diag(LAM);
%     indlam = (dLAM ~= 0);
%     RLAM = sum(indlam);
%     tmp=dLAM(indlam);
%     LAM=real(diag(tmp));
%     U = U(:,indlam);
%     if R > RLAM
%         R = RLAM
%         if outlier == true
%             alpha = sqrt(R)*alpha_s;
%         end
%     end

    tmp=diag(LAM); tmp=tmp(1:R); LAM=diag(tmp);
    U=U(:,1:R);
    Y=real(sqrt(LAM)*U');
    
    %% linear SSC algorithm in RKHS
    fprintf('Running SSC..\n');
    tic
    [Z,A] = SSC(normc(Y),r,affine,alpha,outlier,rho,labels_in);
    
    % Performance on in-sample data
    ACC_in(i_trial)  = 1-computeCE(A,labels_in)
    NMI_in(i_trial) = compute_nmi(labels_in, A)
    Fscore_in(i_trial) = compute_f(labels_in,A')

    % Clustering out-of-sample data
    [B_y, begB_y, enddB_y, mu_Y]  = bases_estimation(Y,A,dsubspace); % bases estimation

    N_out = size(X_out,2);
    Xtest=normc(X_out)';
    
    % map out-of-sample (test) data to RKHS
    switch (kernel)
        case 'gauss'
            D = pdist2(Xtrain,Xtest);
            Ktest = exp(-(D.*D)/2/sig2);
            clear D
        case 'poly'
            Ktest=(Xtrain*Xtest' + b).^d;
        otherwise
            error('Unknown parameter "ktype".')
    end

    K_ones = K_uncentered*ones(N_in,1)/N_in;
    KXtest = I_ONES*(Ktest-K_ones);
    dl=real(1./sqrt(diag(LAM)));
    Y_out = real(diag(dl)*U'*KXtest);

    for l=1:nc
        Y_outm = Y_out - mu_Y(:,l);    % make data zero mean for distance calculation
        BB=B_y(:,begB_y(l):enddB_y(l));
        Yproj = (BB*BB')*Y_outm;
        Dproj = Y_outm - Yproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A] = min(D); % A is a label
    clear D

    %        Performance on out-of-sample data
    ACC_out(i_trial)  = 1-computeCE(A,labels_out)
    NMI_out(i_trial) = compute_nmi(labels_out, A)
    Fscore_out(i_trial) = compute_f(labels_out,A)
    clear A
    time_elapsed(i_trial)=toc(tstart)
end


display('In-sample perfomance:')
mean_ACC_in = mean(ACC_in)
std_ACC_in = std(ACC_in)
mean_NMI_in = mean(NMI_in)
std_NMI_in = std(NMI_in)
mean_Fscore_in = mean(Fscore_in)
std_Fscore_in = std(Fscore_in)

display('Out-of-sample perfomance:')
mean_ACC_out = mean(ACC_out)
std_ACC_out = std(ACC_out)
mean_NMI_out = mean(NMI_out)
std_NMI_out = std(NMI_out)
mean_Fscore_out = mean(Fscore_out)
std_Fscore_out = std(Fscore_out)



