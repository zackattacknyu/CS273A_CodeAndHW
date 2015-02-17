function [ newXDataClass0, newXDataClass1,newYvecClass0, newYvecClass1,maxInfoGain,colNum ] = getDecTreeSplit( xVals,yVec )
%GETDECTREESPLIT Summary of this function goes here
%   Detailed explanation goes here

sizeX = size(xVals);
numVals = size(yVec);
numFeatures = sizeX(2);

probClass1 = size(find(yVec==1))/numVals;

if probClass1 <= 0
    newXDataClass0 = xVals;
    newYvecClass0 = yVec;
    colNum = 0;
   return 
elseif probClass1 >= 1
    newXDataClass1 = xVals;
    newYvecClass1 = yVec;
    colNum = 0;
    return
end

%calculates entropy.
entropy = getEntropy(probClass1);

maxInfoGain=-Inf;

for xCol = 1:numFeatures
    xData = xVals(:,xCol);
    probX1 = size(find(xData==1))/numVals;
    probX0 = 1-probX1;
    yDataForX1 = yVec(xData==1);
    yDataForX0 = yVec(xData==0);

    probClass1ForX1 = size(find(yDataForX1==1))/size(yDataForX1);
    newEntropyX1 = getEntropy(probClass1ForX1);

    probClass1ForX0 = size(find(yDataForX0==1))/size(yDataForX0);
    newEntropyX0 = getEntropy(probClass1ForX0);

    information = probX1*(entropy-newEntropyX1)+probX0*(entropy-newEntropyX0);
    if(information > maxInfoGain)
       maxInfoGain = information;
       colNum = xCol;
       newXDataClass0 = xVals(xData==0,:);
       newXDataClass1 = xVals(xData==1,:);
       newYvecClass0 = yVec(xData==0);
       newYvecClass1 = yVec(xData==1);
    end
end

end

