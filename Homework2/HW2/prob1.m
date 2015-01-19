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
plot(Xtr,Ytr,'g.','MarkerSize',10)
%plot(Xte,Yte,'rx','MarkerSize',10)
legend('Prediction','Training Data','Location','SouthEast');

%calculate MSE
YhatTr = predict(lr,Xtr); %gets predicted y for training data
YhatTe = predict(lr,Xte); %gets predicted y for test data
mseTr = mean(abs(YhatTr-Ytr).^2);
mseTe = mean(abs(YhatTe-Yte).^2);

%%
%Part C

%degs = [1 3 5 7 10 18];
degs = [10 18];
YtrError = zeros(1,length(degs));
YteError = zeros(1,length(degs));
xs = (min(X):.05:max(X))'; % densely sample possible x-values

figure

for i = 1:length(degs)
    
    degree = degs(i);
    
   % create poly features up to given degree; no "1" feature
    XtrP = fpoly(Xtr, degree, false); 

    [XtrP, M,S] = rescale(XtrP); % it's often a good idea to scale the features
    lr = linearRegress( XtrP, Ytr ); % create and train model

    % defines an "implicit function" Phi(x)
    Phi = @(x) rescale( fpoly(x,degree,false), M,S); 

    % parameters "degree", "M", and "S" are memorized at the function definition
    % Now, Phi will do the required feature expansion and rescaling:
    YhatTrain = predict( lr, Phi(Xtr) ); % predict on training data
    YhatTest = predict(lr, Phi(Xte) ); 
    
    ys = predict( lr, Phi(xs) ); % make predictions at xs
    
    subplot(1,2,i)
    plot(Xtr,Ytr,'g.','MarkerSize',10);
    hold on
    plot(Xte,Yte,'rx','MarkerSize',10); 
    plot(xs,ys)
    axis([0 10 -2 7]);
    title(strcat('f(x) degree=',num2str(degree)));
    hold off
    
    %now get the training and test error
    YtrError(i) = sum((YhatTrain-Ytr).^2)/length(Ytr);
    YteError(i) = sum((YhatTest-Yte).^2)/length(Yte);
    
end

%creates the training and test error plots
figure
semilogy(degs,YtrError);
hold on
semilogy(degs,YteError);
legend('Training Error','Test Error','Location','Northwest');









