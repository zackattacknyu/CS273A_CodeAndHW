%%
load('data/iris.txt');
X = iris(:,1:2);
Y = iris(:,5);

%%
plot(X(:,1),X(:,2),'ro')