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