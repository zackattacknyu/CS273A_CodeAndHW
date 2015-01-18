%%
iris=load('data/curve80.txt'); 
y=iris(:,2); 
X=iris(:,1);

%Part A
[Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test

%%
%Part B
lr = linearRegress( Xtr, Ytr ); % create and train model
xs = (0:.05:10)'; % densely sample possible x-values
ys = predict( lr, xs ); % make predictions at xs

plot(xs,ys)
hold on
plot(Xtr,Ytr,'rx')
plot(Xte,Yte,'g.')
legend('Prediction','Training Data','Test Data','Location','SouthEast');

%calculate MSE
YhatTr = predict(lr,Xtr); %gets predicted y for training data
YhatTe = predict(lr,Xte); %gets predicted y for test data
mseTr = sum(abs(YhatTr-Ytr).^2);
mseTe = sum(abs(YhatTe-Yte).^2);

%%
%Part C

%Xtr2 = [Xtr, Xtr.^2];

degree=3;
% create poly features up to given degree; no "1" feature
XtrP = fpoly(Xtr, degree, false); 

[XtrP, M,S] = rescale(XtrP); % it's often a good idea to scale the features
lr = linearRegress( XtrP, Ytr ); % create and train model

% defines an "implicit function" Phi(x)
Phi = @(x) rescale( fpoly(x,degree,false), M,S); 

% parameters "degree", "M", and "S" are memorized at the function definition
% Now, Phi will do the required feature expansion and rescaling:
YhatTrain = predict( lr, Phi(Xtr) ); % predict on training data
YhatTest = predict(lr, Phi(Xte) ); 
xs = (0:.05:10)'; % densely sample possible x-values
ys = predict( lr, Phi(xs) ); % make predictions at xs
plot(xs,ys)
hold on
plot(Xtr,Ytr,'g.');
plot(Xte,Yte,'rx');

%now get the training and test error
YtrError = sum((YhatTrain-Ytr).^2)/length(Ytr);
YteError = sum((YhatTest-Yte).^2)/length(Yte);






