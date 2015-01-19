%%
iris=load('data/curve80.txt'); 
y=iris(:,2); 
X=iris(:,1);
%%
nFolds = 5;
J = zeros(1,nFolds);
for iFold = 1:nFolds,
    
    degree = 1;
    
    % take ith data block as validation
    [Xti,Xvi,Yti,Yvi] = crossValidate(X,y,nFolds,iFold); 
    
    XtrP = fpoly(Xti, degree, false); 

    [XtrP, M,S] = rescale(XtrP); % it's often a good idea to scale the features
    lr = linearRegress( XtrP, Yti ); % create and train model

    % defines an "implicit function" Phi(x)
    Phi = @(x) rescale( fpoly(x,degree,false), M,S); 

    % parameters "degree", "M", and "S" are memorized at the function definition
    % Now, Phi will do the required feature expansion and rescaling:
    YhatTest = predict(lr, Phi(Xvi) );
    
    J(iFold) = sum((YhatTest-Yvi).^2)/length(Yvi);
end;
% the overall estimated validation performance is the average of the performance on each fold
mean(J)