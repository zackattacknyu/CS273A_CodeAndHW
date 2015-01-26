iris=load('data/iris.txt'); % load the text file
X = iris(:,1:2); Y=iris(:,end); % get first two features
[X Y] = shuffleData(X,Y); % reorder randomly
X = rescale(X); % works much better for rescaled data

XA = X(Y<2,:); YA=Y(Y<2); % get class 0 vs 1
XB = X(Y>0,:); YB=Y(Y>0); % get class 1 vs 2

%%
%part A

%plot class 0 and class 1
figure
scatter(XA(:,1),XA(:,2),20,YA); %plot the data points
colormap winter;
colorbar

%plot class 1 and class 2
figure
scatter(XB(:,1),XB(:,2),20,YB); %plot the data points
colormap winter;
colorbar

%%
%Part B

%weights to demo for part b
wts = [0.5 1 -0.25];

%define the two learners we will use
learnerA=logisticClassify2();
learnerB=logisticClassify2();

%set the class labels for our learners
learnerA=setClasses(learnerA, unique(YA));
learnerB=setClasses(learnerB, unique(YB));

%sets the weights for both learners
learnerA=setWeights(learnerA, wts);
learnerB=setWeights(learnerB, wts);

%plot the data and the decision boundary
plot2DLinear(learnerA,XA,YA);
plot2DLinear(learnerB,XB,YB);

%%
%part C

%get the predicted class labels for A and B
YhatA = predict(learnerA,XA);
YhatB = predict(learnerB,XB);

%get the error rate
errorA = length(find(YhatA~=YA))/length(YA);
errorB = length(find(YhatB~=YB))/length(YB);

%%
%Part E

theta = [0.5 1 -0.25];
alpha = 0;
stepSize = 0.1;

for j = 1:length(Y)
    
   %data that depends on our particular point
    yj = Y(j);
    xj = [1 X(j,:)];
    zValue = dot(theta,xj);
    expZ = exp(-zValue);

    %calculate J'
    JjPrime = zeros(1,length(theta));
    for i = 1:length(theta)
        JjPrime(i) = (1-yj)*xj(i)*theta(i) - expZ*xj(i)*theta(i)/(1+expZ) ...
            + 2*theta(i)*alpha;
    end

    %do the gradient step
    theta = theta - JjPrime.*stepSize; 
    
end

