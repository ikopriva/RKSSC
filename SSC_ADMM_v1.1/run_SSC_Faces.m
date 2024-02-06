%--------------------------------------------------------------------------
% This is the main function to run the SSC algorithm for the face 
% clustering problem on the Extended Yale B dataset.
% avgmissrate: the n-th element contains the average clustering error for 
% n subjects
% medmissrate: the n-th element contains the median clustering error for 
% n subjects
% nSet: the set of the different number of subjects to run the algorithm on
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

clear all, close all

load YaleBCrop025.mat

alpha = 20;

nSet = [2 3 5 8 10];
for i = 1:length(nSet)
    n = nSet(i);
    idx = Ind{n};   
    for j = 1:size(idx,1)
        X = [];
        for p = 1:n
            X = [X Y(:,:,idx(j,p))];
        end
        [D,N] = size(X);
       
        r = 0; affine = false; outlier = true; rho = 1;
        [missrate,C] = SSC(X,r,affine,alpha,outlier,rho,s{n});
        missrateTot_SSC{n}(j) = missrate;
        
        % ELRASSC
        tau=1e0; lam_elra=1e-4;
        Xp = DataProjection(X,r);
        nn = max(s{n});

       % Xp=X;
        alpha_elrassc = [2 20 20];
        %[C] = admm_elrassc_func(Xp,alpha_elrassc,tau,lam_elra,2e-4,200);
        %CKSym = BuildAdjacency(thrC(C,rho));
        lam_0 = 1e1; lam_1 = 1e2; c = 0.5; mu = 1.5; thr = 1e-8; maxIter = 200; 
        kernel = 'linear'; var1 = 1e3; var2 = [];
        CKSym = admm_elrsa_sc_func(Xp,lam_0, lam_1,c, mu,thr, maxIter, kernel, var1, var2);
        grps = SpectralClustering(CKSym,nn);
        missrate = Misclassification(grps,s{n});
        missrateTot_ELRASSC{n}(j) = missrate;

        save SSC_Faces.mat missrateTot_SSC missrateTot_ELRASSC alpha
    end
    avgmissrate_SSC(n) = mean(missrateTot_SSC{n});
    medmissrate_SSC(n) = median(missrateTot_SSC{n});
    avgmissrate_ELRASSC(n) = mean(missrateTot_ELRASSC{n});
    medmissrate_ELRASSC(n) = median(missrateTot_ELRASSC{n});
    
    save SSC_Faces.mat missrateTot_SSC missrateTot_ELRASSC avgmissrate_SSC avgmissrate_ELRASSC medmissrate_SSC medmissrate_ELRASSC alpha
end

save SSC_Faces.mat missrateTot_SSC missrateTot_ELRASSC avgmissrate_SSC avgmissrate_ELRASSC medmissrate_SSC medmissrate_ELRASSC alpha