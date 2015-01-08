iris = load('HW1/data/iris.txt');
y = iris(:,end);
X = iris(:,1:end-1);

%part A
numFeatures = size(X,2);
numDataPoints = size(X,1);

%part B
figure
subplot(2,2,1)
hist(X(:,1))
title('Histogram for Feature 1')
subplot(2,2,2)
hist(X(:,2))
title('Histogram for Feature 2')
subplot(2,2,3)
hist(X(:,3))
title('Histogram for Feature 3')
subplot(2,2,4)
hist(X(:,4))
title('Histogram for Feature 4')
