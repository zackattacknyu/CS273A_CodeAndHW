iris=load('data/iris.txt'); 
y=iris(:,end); 
X=iris(:,1:end-1);

[X y] = shuffleData(X,y); % shuffle data randomly
[Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test

%gets the first 2 features
XtrFirstTwo = Xtr(:,1:2);
XteFirstTwo = Xte(:,1:2);

%train the classifier
K = 3;
knn = knnClassify( XtrFirstTwo, Ytr, K ); % replace or set K to some integer
YteHat = predict( knn, XteFirstTwo ); % make predictions on Xtest

plotClassify2D( knn, XtrFirstTwo, Ytr ); % make 2D classification plot with data (Xtr,Ytr)