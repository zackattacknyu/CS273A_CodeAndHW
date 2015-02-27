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
[z,c,~] = kmeans(X,k);

figure
scatter(X(:,1),X(:,2),20,z); %plot the data points
colorbar
hold on
plot(c(:,1),c(:,2),'rx');
