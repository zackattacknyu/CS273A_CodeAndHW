%%
Xte = load('kaggle/kaggle.X1.test.txt');
Xtr = load('kaggle/kaggle.X1.train.txt');
Ytr = load('kaggle/kaggle.Y.train.txt');

%%

%Part A
[Xtrain,Xvalid,Ytrain,Yvalid] = splitData(Xtr,Ytr,0.8);
dt = treeRegress(Xtrain,Ytrain, 'maxDepth',20);
mseTree = mse(dt,Xvalid,Yvalid);

%%

%Part B
validMSE = ones(1,16);
trainMSE = ones(1,16);
index = 1;
for depth = 0:15
    dt = treeRegress(Xtrain,Ytrain, 'maxDepth',depth);
    trainMSE(index) = mse(dt,Xtrain,Ytrain);
    validMSE(index) = mse(dt,Xvalid,Yvalid);
    index = index + 1;
end

plot(0:15,validMSE,'g-')
hold on
plot(0:15,trainMSE','r--')
xlabel('MaxDepth Value');
ylabel('MSE');
title('MSE versus MaxDepth');
legend('Validation MSE','Training MSE');
[minMSE,indexOfMinMSE] = min(validMSE);
depthOfMinMSE = indexOfMinMSE + 1;

%%

%Part C

minParentVals = 2.^(3:12);
validMSE2 = zeros(1,10);
trainMSE2 = zeros(1,10);
for val = 1:10
    dt = treeRegress(Xtrain,Ytrain,'maxDepth',20,'minParent',minParentVals(val));
    trainMSE2(val) = mse(dt,Xtrain,Ytrain);
    validMSE2(val) = mse(dt,Xvalid,Yvalid);
end

plot(3:12,validMSE2,'g-');
hold on
plot(3:12,trainMSE2,'r--');
xlabel('log_2(MinParent) Value');
ylabel('MSE');
title('MSE versus minParent');
legend('Validation MSE','Training MSE');
[minMSE2,indexOfMinMSE2] = min(validMSE2);

%%

%Part D

%Call 1: Kaggle score was 0.65140
%dt = treeRegress(Xtr,Ytr,'maxDepth',9,'minParent',2^9);

%Call 2: Kaggle score was 0.64843, the best one
dt = treeRegress(Xtr,Ytr,'maxDepth',20,'minParent',2^9);

%Call 3: Kaggle score was 0.66217
%dt = treeRegress(Xtr,Ytr,'maxDepth',9); 

Yhat = predict(dt,Xte);

fh = fopen('kagglePrediction.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(Yhat),
    fprintf(fh,'%d,%d\n',i,Yhat(i));  % output each prediction
end;
fclose(fh);                         % close the file