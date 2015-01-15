%%
iris=load('data/curve80.txt'); 
y=iris(:,2); 
X=iris(:,1);

%Part A
[Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test

%%
%Part B
lr = linearRegress( Xtr, Ytr ); % create and train model
xs = (0:.05:10)'; % densely sample possible x-values
ys = predict( lr, xs ); % make predictions at xs

plot(xs,ys)
hold on
plot(Xtr,Ytr,'rx')
plot(Xte,Yte,'g.')

%calculate MSE
YhatTr = predict(lr,Xtr); %gets predicted y for training data
YhatTe = predict(lr,Xte); %gets predicted y for test data
mseTr = sum(abs(YhatTr-Ytr).^2);
mseTe = sum(abs(YhatTe-Yte).^2);