iris = load('data/iris.txt');
y = iris(:,end);
X = iris(:,1:end-1);

%part A
numFeatures = size(X,2);
numDataPoints = size(X,1);

%put features into vectors
feature1 = X(:,1);
feature2 = X(:,2);
feature3 = X(:,3);
feature4 = X(:,4);

%part B
figure
subplot(2,2,1)
hist(feature1)
title('Histogram for Feature 1')
subplot(2,2,2)
hist(feature2)
title('Histogram for Feature 2')
subplot(2,2,3)
hist(feature3)
title('Histogram for Feature 3')
subplot(2,2,4)
hist(feature4)
title('Histogram for Feature 4')

%part C
mean1 = mean(feature1);
mean2 = mean(feature2);
mean3 = mean(feature3);
mean4 = mean(feature4);


%part D

%compute the variance
var1 = var(feature1);
var2 = var(feature2);
var3 = var(feature3);
var4 = var(feature4);

%compute the standard deviation
std1 = std(feature1);
std2 = std(feature2);
std3 = std(feature3);
std4 = std(feature4);

%part E
% Normalizes the data
normalize1 = (feature1-mean1)/std1;
normalize2 = (feature2-mean2)/std2;
normalize3 = (feature3-mean3)/std3;
normalize4 = (feature4-mean4)/std4;

%part F
size = 30;
figure
subplot(1,3,1)
plot(normalize1(y==0),normalize2(y==0),'o');
hold all
plot(normalize1(y==1),normalize2(y==1),'o');
plot(normalize1(y==2),normalize2(y==2),'o');
legend('y=0','y=1','y=2','Location','SouthEast');
title('Feature 1 vs. Feature 2');
xlabel('Feature 1 Value');
ylabel('Feature 2 Value');
hold off
subplot(1,3,2)
plot(normalize1(y==0),normalize3(y==0),'o');
hold all
plot(normalize1(y==1),normalize3(y==1),'o');
plot(normalize1(y==2),normalize3(y==2),'o');
legend('y=0','y=1','y=2','Location','SouthEast');
title('Feature 1 vs. Feature 3');
xlabel('Feature 1 Value');
ylabel('Feature 3 Value');
hold off
subplot(1,3,3)
plot(normalize1(y==0),normalize4(y==0),'o');
hold all
plot(normalize1(y==1),normalize4(y==1),'o');
plot(normalize1(y==2),normalize4(y==2),'o');
legend('y=0','y=1','y=2','Location','SouthEast');
title('Feature 1 vs. Feature 4');
xlabel('Feature 1 Value');
ylabel('Feature 4 Value');
hold off


















