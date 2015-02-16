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

yVec = xyData(:,6);
numVals = size(yVec);

probClass1 = size(find(yVec==1))/numVals;

%calculates entropy.
entropy = getEntropy(probClass1);

information = zeros(1,5);

for xCol = 1:5
    xData = xyData(:,xCol);
    probX1 = size(find(xData==1))/numVals;
    probX0 = 1-probX1;
    yDataForX1 = yVec(xData==1);
    yDataForX0 = yVec(xData==0);

    probClass1ForX1 = size(find(yDataForX1==1))/size(yDataForX1);
    newEntropyX1 = getEntropy(probClass1ForX1);

    probClass1ForX0 = size(find(yDataForX0==1))/size(yDataForX0);
    newEntropyX0 = getEntropy(probClass1ForX0);

    information(xCol) = probX1*(entropy-newEntropyX1)+probX0*(entropy-newEntropyX0);
end

[gain,colNum] = max(information);

%make sure function works
[newXDataClass0, newXDataClass1,newYvecClass0, newYvecClass1...
    ,maxInfoGain,colNum ] = getDecTreeSplit( xyData(:,1:5),yVec );

%we split on class 2

%now we split x_2=0
[newXData2Class0, newXData2Class1,newYvec2Class0, newYvec2Class1...
    ,maxInfoGain2,colNum2 ] = getDecTreeSplit( newXDataClass0,newYvecClass0 );

%now we split x_2=1
[newXData3Class0, newXData3Class1,newYvec3Class0, newYvec3Class1...
    ,maxInfoGain3,colNum3 ] = getDecTreeSplit( newXDataClass1,newYvecClass1 );