%%
iris=load('data/curve80.txt'); 
y=iris(:,2); 
X=iris(:,1);

[Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test

%%
degs = [1 3 5 7 10 18];
crossValidError = zeros(1,length(degs));

nFolds = 5;
J = zeros(1,nFolds);

for k=1:length(degs)
    
    degree = degs(k);
    
    for iFold = 1:nFolds,

        % take ith data block as validation
        [Xti,Xvi,Yti,Yvi] = crossValidate(Xtr,Ytr,nFolds,iFold); 

        XtiP = fpoly(Xti, degree, false); 

        [XtiP, M,S] = rescale(XtiP); % it's often a good idea to scale the features
        lr = linearRegress( XtiP, Yti ); % create and train model
        XviP = rescale( fpoly(Xvi,degree,false), M,S); 
        YhatTest = predict(lr, XviP );

        J(iFold) = mean((YhatTest-Yvi).^2);
    end;
    
    % the overall estimated validation performance is the average of the performance on each fold
    crossValidError(k) = mean(J);
end

semilogy(degs,crossValidError,'b-','LineWidth',2);
xlabel('Polynomial Degree');
ylabel('Average cross-validation MSE');


