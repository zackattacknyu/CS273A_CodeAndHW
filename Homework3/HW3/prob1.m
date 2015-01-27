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
figure
plot2DLinear(learnerA,XA,YA);
figure
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
learnerA=logisticClassify2();
learnerA=train(learnerA,XA,YA,'stopIter',100,'stopTol',0.0001);

%%
%Part E practice script
theta = [0.5 1 -0.25];
alpha = 0.2;
stepSize = 0.02;

%define the learner used for plotting
learnerGradDesc=logisticClassify2();
learnerGradDesc=setClasses(learnerGradDesc, unique(YA));

%loss values
Jj = zeros(1,length(YA));
minLoss = 100;
bestTheta = theta;
for j = 1:length(YA)
    
   %data that depends on our particular point
    yj = YA(j);
    xj = [1 XA(j,:)];
    zValue = dot(theta,xj);
    sigmaZ = 1/(1+exp(-zValue));

    %calculate J'
    JjPrime = zeros(1,length(theta));
    for i = 1:length(theta)
        JjPrime(i) = xj(i) * (sigmaZ - yj) + 2*theta(i)*alpha;
    end

    %do the gradient step
    theta = theta - JjPrime.*stepSize;
    
    %find the loss function value
    for k = 1:length(YA)
        yk = YA(k);
        xk = [1 XA(k,:)];
        zValueK = dot(theta,xk);
        sigmaZk = 1/(1+exp(-zValueK));
        Jj(j) = Jj(j) + -yk*log(sigmaZk) + (1-yk)*log(1-sigmaZk) ...
            + alpha*sum(theta.^2);
    end
    
    if(Jj(j) < minLoss)
       minLoss = Jj(j);
       bestTheta = theta;
    end
    
    
    %replot the data
    %learnerGradDesc=setWeights(learnerGradDesc, theta);
    %plot2DLinear(learnerGradDesc,XA,YA);
    
    %pause to see the new weights
    %pause(0.2);
end

learnerGradDesc=setWeights(learnerGradDesc, bestTheta);
plot2DLinear(learnerGradDesc,XA,YA);
