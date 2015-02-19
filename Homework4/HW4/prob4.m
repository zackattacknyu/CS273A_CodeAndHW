Xte = load('kaggle/kaggle.X1.test.txt');
Xtr = load('kaggle/kaggle.X1.train.txt');
Ytr = load('kaggle/kaggle.Y.train.txt');

%%
%start with "mean" predictor
mu = mean(Ytr); 
curY = Ytr - mu; 

%number of ensembles
N = 5;

%alpha values
alpha = ones(1,N);
dt = cell(1,N);

for k=1:N,
 dt{k} = treeRegress(Xtr,curY,'maxDepth',3);
 curY = curY - alpha(k) * predict(dt{k}, Xtr);
end;
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