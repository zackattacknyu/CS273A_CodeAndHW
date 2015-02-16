iris=load('data/iris.txt'); % load the text file
X = iris(:,1:2); Y=iris(:,end); % get first two features

% get class 0 vs 1
XA = X(Y<2,:); YA=Y(Y<2); 
sizeXA = size(YA);
numFeatures = sizeXA(1);

Yvar = YA.*2 - 1; %turns 0 and 1 into -1 and 1

%variables for quadprog
H = [0 0 0; 0 2 0;0 0 2];
f = [0 0 0];

%gets the A matrix
constCol = ones(numFeatures,1);
Ainit = [constCol XA];
A = ones(numFeatures,3);
for row = 1:numFeatures
   A(row,:) = Ainit(row,:).*(Yvar(row)*-1);
end

%gets the b column
b = ones(numFeatures,1).*-1;

theta = quadprog(H,f,A,b);

%plots the classification boundary from this
learnerA=logisticClassify();
learnerA=setClasses(learnerA, unique(YA));
learnerA=setWeights(learnerA, theta');
%YhatA = predict(learnerA,XA);
%figure
plotClassify2D(learnerA,XA,YA);
