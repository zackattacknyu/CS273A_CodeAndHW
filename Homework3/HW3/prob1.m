iris=load('data/iris.txt'); % load the text file
X = iris(:,1:2); Y=iris(:,end); % get first two features
[X Y] = shuffleData(X,Y); % reorder randomly
X = rescale(X); % works much better for rescaled data
XA = X(Y<2,:); YA=Y(Y<2); % get class 0 vs 1
XB = X(Y>0,:); YB=Y(Y>0); % get class 1 vs 2
