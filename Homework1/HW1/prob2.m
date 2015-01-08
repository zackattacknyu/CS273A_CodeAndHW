%InitialPart
iris=load('data/iris.txt'); 
y=iris(:,end); 
X=iris(:,1:end-1);

[X y] = shuffleData(X,y); % shuffle data randomly
[Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test

%gets the first 2 features
XtrFirstTwo = Xtr(:,1:2);
XteFirstTwo = Xte(:,1:2);

%knn = knnClassify( XtrFirstTwo, Ytr, K ); % replace or set K to some integer
%YteHat = predict( knn, XteFirstTwo ); % make predictions on Xtest

%plotClassify2D( knn, XtrFirstTwo, Ytr ); % make 2D classification plot with data (Xtr,Ytr)

%partA
%{

figure
Kvals = [1,5,10,50];
for i=1:4
   K = Kvals(i);
   
   %train the classifier
   knn = knnClassify( XtrFirstTwo, Ytr, K );
   
   % make 2D classification plot
   subplot(2,2,i)
   plotClassify2D( knn, XtrFirstTwo, Ytr );
   title(strcat('K=',num2str(K),' classification plot'));
end

%}
%part B

K = 50;
learner = knnClassify( XtrFirstTwo, Ytr, K );
Yhat = predict(learner,XtrFirstTwo);
errTrain=length(find(Yhat~=Ytr))/length(Ytr);
%errTrain(i) = ... % TODO: " " to count what fraction of predictions are wrong

%{
Kvals=[1,2,5,10,50,100,200];
for i=1:length(Kvals)
    K = Kvals(i);
    learner = knnClassify( XtrFirstTwo, Ytr, K );
    Yhat = predict(learner,XtrFirstTwo);
    errTrain(i) = ... % TODO: " " to count what fraction of predictions are wrong
    %TODO: repeat prediction / error evaluation for test data
end;
figure; semilogx(... % TODO: " " to average and plot results on semi-log scale

%}