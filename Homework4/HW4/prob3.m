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
index = 1;
for depth = 0:15
    dt = treeRegress(Xtrain,Ytrain, 'maxDepth',depth);
    validMSE(index) = mse(dt,Xvalid,Yvalid);
    index = index + 1;
end

plot(validMSE)
[minMSE,indexOfMinMSE] = min(validMSE);
depthOfMinMSE = indexOfMinMSE + 1;

%%

%Part C

minParentVals = 2.^(3:12);
validMSE2 = zeros(1,10);
for val = 1:10
    dt = treeRegress(Xtrain,Ytrain,'maxDepth',20,'minParent',minParentVals(val));
   validMSE2(val) = mse(dt,Xvalid,Yvalid);
end

plot(validMSE2)
[minMSE2,indexOfMinMSE2] = min(validMSE2);

%%

%Part D
%dt = treeRegress(Xtr,Ytr,'maxDepth',9,'minParent',2^9);
%dt = treeRegress(Xtr,Ytr,'maxDepth',20,'minParent',2^9);
dt = treeRegress(Xtr,Ytr,'maxDepth',9);
Yhat = predict(dt,Xte);

%%
fh = fopen('kagglePrediction.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(Yhat),
    fprintf(fh,'%d,%d\n',i,Yhat(i));  % output each prediction
end;
fclose(fh);                         % close the file