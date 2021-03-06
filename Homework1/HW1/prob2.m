%%
%InitialPart
iris=load('data/iris.txt'); 
y=iris(:,end); 
X=iris(:,1:end-1);

[X y] = shuffleData(X,y); % shuffle data randomly
[Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test

%gets the first 2 features
XtrFirstTwo = Xtr(:,1:2);
XteFirstTwo = Xte(:,1:2);

%%
%partA
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
%%
%part B
Kvals=[1,2,5,10,50,100,200];
errTrain=zeros(1,length(Kvals));
errTest = zeros(1,length(Kvals));
for i=1:length(Kvals)
    K = Kvals(i);
    learner = knnClassify( XtrFirstTwo, Ytr, K );
    YhatTr = predict(learner,XtrFirstTwo);
    errTrain(i)=length(find(YhatTr~=Ytr))/length(Ytr);
    
    YhatTe = predict(learner,XteFirstTwo);
    errTest(i)=length(find(YhatTe~=Yte))/length(Yte);
end;

figure
semilogx(Kvals,errTrain,'r-');
hold on
semilogx(Kvals,errTest,'g-');
legend('Training Error','Test Error','Location','SouthEast');
