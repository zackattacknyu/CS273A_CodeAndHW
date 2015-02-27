%%
load('data/iris.txt');
X = iris(:,1:2);
Y = iris(:,5);

%%

%Part A
plot(X(:,1),X(:,2),'ro')

%%

%Part B
k=5;
%k=20;

%make 5 initial points with star shape
minX1 = min(X(:,1)); 
minX2 = min(X(:,2));
maxX1 = max(X(:,1));
maxX2 = max(X(:,2));
centerX1 = (minX1 + maxX1)/2; centerX2 = (minX2 + maxX2)/2;
firstQuatX1 = minX1 + (maxX1-minX1)/4;
thirdQuatX1 = maxX1 - (maxX1-minX1)/4;
firstQuatX2 = minX2 + (maxX2-minX2)/4;
thirdQuatX2 = maxX2 - (maxX2-minX2)/4;
centerPt = [centerX1 centerX2];
Pt11 = [firstQuatX1 firstQuatX2];
Pt13 = [firstQuatX1 thirdQuatX2];
Pt31 = [thirdQuatX1 firstQuatX2];
Pt33 = [thirdQuatX1 thirdQuatX2];
initPts = [centerPt;Pt11;Pt13;Pt31;Pt33];

%run k-Means with the different initializations
[z1,c1,score1] = kmeans(X,k,'random');
[z2,c2,score2] = kmeans(X,k,'farthest');
[z3,c3,score3] = kmeans(X,k,'k++');
[z4,c4,score4] = kmeans(X,k,initPts);

%%
figure
plotClassify2D([],X,z)
hold on
plot(c(:,1),c(:,2),'rx');

%%


