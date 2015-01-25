iris=load('data/iris.txt'); % load the text file
X = iris(:,1:2); Y=iris(:,end); % get first two features
[X Y] = shuffleData(X,Y); % reorder randomly
X = rescale(X); % works much better for rescaled data

%part A
X0 = X(Y==0,:); Y0=Y(Y==0);
X1 = X(Y==1,:); Y1=Y(Y==1);
X2 = X(Y==2,:); Y2=Y(Y==2);

%%
%plot class 0 and class 1
figure
plot(X0(:,1),X0(:,2),'r.');
hold on
plot(X1(:,1),X1(:,2),'gx');
legend('Class 0','Class 1');
hold off

%plot class 1 and class 2
figure
plot(X1(:,1),X1(:,2),'gx');
hold on
plot(X2(:,1),X2(:,2),'bo');
legend('Class 1','Class 2','Location','SouthEast');
hold off

%%
%Part B
XA = X(Y<2,:); YA=Y(Y<2); % get class 0 vs 1
XB = X(Y>0,:); YB=Y(Y>0); % get class 1 vs 2

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