Xte = load('kaggle/kaggle.X1.test.txt');
Xtr = load('kaggle/kaggle.X1.train.txt');
Ytr = load('kaggle/kaggle.Y.train.txt');

%%

[Xtrain,Xvalid,Ytrain,Yvalid] = splitData(Xtr,Ytr,0.8);

%%
%start with "mean" predictor
mu = mean(Ytrain); 
curY = Ytrain - mu; 

%number of ensembles
N = 25;

mseTraining = zeros(1,N);
mseValidation = zeros(1,N);

%alpha values
alpha = ones(1,N);
dt = cell(1,N);

[Nvalid,Dval] = size(Xvalid);

predictY = zeros(Nvalid,1); % Allocate space

for k=1:N,
 dt{k} = treeRegress(Xtrain,curY,'maxDepth',3);
 curY = curY - alpha(k) * predict(dt{k}, Xtrain);
 
 %find training MSE at k
 mseTraining(k) = mean((curY-Ytrain).^2);
 
 %find validation MSE
 predictY = predictY + alpha(k)*predict(dt{k}, Xvalid);
 mseValidation(k) = mean((Yvalid-predictY).^2);
 
end;

%%

plot(mseTraining,'r-');
hold on
plot(mseValidation,'g--');
%%
% Test data Xtest
[Nte,D] = size(Xte);
predictY = zeros(Nte,1); % Allocate space
for k=1:N, % Predict with each learner
 predictY = predictY + alpha(k)*predict(dt{k}, Xte);
end; 

%%
fh = fopen('kagglePrediction.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(predictY),
    fprintf(fh,'%d,%d\n',i,predictY(i));  % output each prediction
end;
fclose(fh);                         % close the file