xyData=[
0 0 1 1 0 -1;
1 1 0 1 0 -1;
0 1 1 1 1 -1;
1 1 1 1 0 -1;
0 1 0 0 0 -1;
1 0 1 1 1 1;
0 0 1 0 0 1;
1 0 0 0 0 1;
1 0 1 1 0 1;
1 1 1 1 1 -1];   

X=xyData(:,1:5);
y=xyData(:,6);

%Part A

%this gets p(y==1) and p(y==-1)
indicesY1 = find(y==1);
indicesYminus1 = find(y==-1);
probY1 = length(indicesY1)/length(y);
probYminus1 = 1-probY1;

%in order to get class probabilities, we go over entries where
%       y==1 and y==-1
%
% We will get the following matrices
%   probXwhereY1:
%       row i has probabilities for x_i
%       column 1 is p(x_i=0|y=1)
%       column 2 is p(x_i=1|y=1)
%   probXwhereYminus1:
%       row i has probabilities for x_i
%       column 1 is p(x_i=0|y=-1)
%       column 2 is p(x_i=1|y=-1)
probXwhereY1 = zeros(5,2);
probXwhereYminus1 = zeros(5,2);
for i = 1:5
    Xi = X(:,i);
    
    XiwhereY1 = Xi(indicesY1); %x_i values for entries where y=1
    probXwhereY1(i,1) = length(find(XiwhereY1==0))/length(XiwhereY1);
    probXwhereY1(i,2) = 1-probXwhereY1(i,1);
    
    XiwhereYminus1 = Xi(indicesYminus1); %x_i values for entries where y=-1
    probXwhereYminus1(i,1) = length(find(XiwhereYminus1==0))/length(XiwhereYminus1);
    probXwhereYminus1(i,2) = 1-probXwhereYminus1(i,1);
end

%part B
%Using Naive Bayes, p(y|x) = p(x|y)p(y)/p(x)
%   p(x) = sum_y( p(x|y)p(y) )
%   p(x|y)=p(x_1|y)p(x_2|y)...p(x_5|y)

%We need to find the y prob of x=(0 0 0 0 0)
xTest1 = [0 0 0 0 0];
[probY1withXtest1,probYminus1withXtest1] = prob3Classifier(probXwhereY1,probXwhereYminus1,probY1,probYminus1,xTest1);

bestYclassification1 = 1;
if(probY1withXtest1<probYminus1withXtest1)
    bestYclassification1 = -1; 
end

xTest2 = [1 1 0 1 0];
[probY1withXtest2,probYminus1withXtest2] = prob3Classifier(probXwhereY1,probXwhereYminus1,probY1,probYminus1,xTest2);

bestYclassification2 = 1;
if(probY1withXtest1<probYminus1withXtest1)
    bestYclassification2 = -1; 
end


