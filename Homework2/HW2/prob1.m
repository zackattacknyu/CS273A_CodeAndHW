%%
%InitialPart
iris=load('data/curve80.txt'); 
y=iris(:,2); 
X=iris(:,1);

[Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test


