function [ probY1withXtest1,probYminus1withXtest1 ] = prob3Classifier( probXwhereY1,probXwhereYminus1, probY1, probYminus1,xTest1 )
%PROB3CLASSIFIER says the probabilities of the classifications of the test
%               data
% Input is the following matrices:
%   probXwhereY1:
%       row i has probabilities for x_i
%       column 1 is p(x_i=0|y=1)
%       column 2 is p(x_i=1|y=1)
%   probXwhereYminus1:
%       row i has probabilities for x_i
%       column 1 is p(x_i=0|y=-1)
%       column 2 is p(x_i=1|y=-1)
%
%probY1withXtest1 is p(y=1|x) where x is the test vector
%probYminus1withXtest1 is p(y=-1|x) where x is the test vector

probXtestWhereY1 = zeros(1,5);
probXtestWhereYminus1 = zeros(1,5);
for i = 1:5
   probXtestWhereY1(i) = probXwhereY1(i,xTest1(i)+1); 
   probXtestWhereYminus1(i) = probXwhereYminus1(i,xTest1(i)+1); 
end
probXtestWithY1 = prod(probXtestWhereY1)*probY1;
probXtestWithYminus1 = prod(probXtestWhereYminus1)*probYminus1;

%finally here is p(y=1|x)
probY1withXtest1 = probXtestWithY1/(probXtestWithY1+probXtestWithYminus1);

%here is p(y=-1|x)
probYminus1withXtest1 = probXtestWithYminus1/(probXtestWithY1+probXtestWithYminus1);

end

