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

minX1 = min(X(:,1)); 
minX2 = min(X(:,2));
maxX1 = max(X(:,1));
maxX2 = max(X(:,2));

%if k=5, make 5 initial points arranged in X-shape
if(k==5)
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
    initPts5 = [centerPt;Pt11;Pt13;Pt31;Pt33];
end

%if k=20, make 20 points in 4x5 and 5x4 arrangement
if(k==20)
    fifthX1 = (maxX1-minX1)/5;
    sixthX1 = (maxX1-minX1)/6;
    fifthX2 = (maxX2-minX2)/5;
    sixthX2 = (maxX2-minX2)/6;
    initPts20A = zeros(20,2);
    initPts20B = zeros(20,2);
    index = 1;
    for i = 1:5
       for j = 1:4
          curX1A = minX1 + fifthX1*j;
          curX2A = minX2 + sixthX2*i;
          initPts20A(index,:) = [curX1A curX2A];

          curX1B = minX1 + sixthX1*i;
          curX2B = minX2 + fifthX2*j;
          initPts20B(index,:) = [curX1B curX2B];

          index = index+1;
       end
    end 
end

%%
%run k-Means with the different initializations
[z1,c1,score1] = kmeans(X,k,'random');
[z2,c2,score2] = kmeans(X,k,'farthest');
[z3,c3,score3] = kmeans(X,k,'k++');

if(k==5)
   [z4,c4,score4] = kmeans(X,k,initPts5); 
end

if(k==20)
    [z5,c5,score5] = kmeans(X,k,initPts20A);
    [z6,c6,score6] = kmeans(X,k,initPts20B);
end


%%
figure
plotClassify2D([],X,z)
hold on
plot(c(:,1),c(:,2),'rx');

%%

%hierarchical aggolomorative clustering
k=5;
z = agglomCluster(X,k,'min');%single linkage
plotClassify2D([],X,z)

z = agglomCluster(X,k,'max');%complete linkage
plotClassify2D([],X,z)

%%
k=5;
[z,T,~,~] = emCluster(X,k);
plotClassify2D([],X,z)
hold on
for i=1:k
   plotGauss2D(T.mu(i,:),T.Sig(:,:,i),'r'); 
end

