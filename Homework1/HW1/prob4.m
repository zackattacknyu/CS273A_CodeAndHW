%InitialPart
iris=load('data/iris.txt'); 
y=iris(:,end); 
X=iris(:,1:end-1);

[X y] = shuffleData(X,y); % shuffle data randomly
[Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test

%gets the first 2 features
XtrFirstTwo = Xtr(:,1:2);
XteFirstTwo = Xte(:,1:2);

%Part A

%classes are 0,1,2 for Y
%
%The indices giving y=0,1, and 2 for training data
Ytr0Indices = find(Ytr==0);
Ytr1Indices = find(Ytr==1);
Ytr2Indices = find(Ytr==2);

XtrClass0 = XtrFirstTwo(Ytr0Indices,:);
XtrClass1 = XtrFirstTwo(Ytr1Indices,:);
XtrClass2 = XtrFirstTwo(Ytr2Indices,:);

XtrMeanClass0 = mean(XtrClass0);
XtrMeanClass1 = mean(XtrClass1);
XtrMeanClass2 = mean(XtrClass2);

XtrCovClass0 = cov(XtrClass0);
XtrCovClass1 = cov(XtrClass1);
XtrCovClass2 = cov(XtrClass2);

%Part B
size = 30;
figure
scatter(XtrFirstTwo(:,1),XtrFirstTwo(:,2),size,Ytr);

%Part C
figure
hold on
scatter(XtrFirstTwo(:,1),XtrFirstTwo(:,2),size,Ytr);
plotGauss2D(XtrMeanClass0,XtrCovClass0,'b-');
plotGauss2D(XtrMeanClass1,XtrCovClass1,'g-');
plotGauss2D(XtrMeanClass2,XtrCovClass2,'y-');
hold off

%Part D
bc = gaussBayesClassify( XtrFirstTwo, Ytr );
figure
plotClassify2D(bc, XtrFirstTwo, Ytr);

%
%Part E
yTrHat = predict(bc,XtrFirstTwo);
trainError = length(find(yTrHat~=Ytr))/length(Ytr);

yTeHat = predict(bc,XteFirstTwo);
testError = length(find(yTeHat~=Yte))/length(Yte);

%%
%Part F
XtrAllClass0 = Xtr(Ytr0Indices,:);
XtrAllClass1 = Xtr(Ytr1Indices,:);
XtrAllClass2 = Xtr(Ytr2Indices,:);

XtrAllMeanClass0 = mean(XtrAllClass0);
XtrAllMeanClass1 = mean(XtrAllClass1);
XtrAllMeanClass2 = mean(XtrAllClass2);

XtrAllCovClass0 = cov(XtrAllClass0);
XtrAllCovClass1 = cov(XtrAllClass1);
XtrAllCovClass2 = cov(XtrAllClass2);

bcTrAll = gaussBayesClassify( Xtr, Ytr );
yTrAllHat = predict(bcTrAll,Xtr);
trainErrorAll = length(find(yTrAllHat~=Ytr))/length(Ytr);

yTeAllHat = predict(bcTrAll,Xte);
testErrorAll = length(find(yTeAllHat~=Yte))/length(Yte);