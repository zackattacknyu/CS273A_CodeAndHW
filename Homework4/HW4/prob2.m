%%
%sets up the data
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
xVals = xyData(:,1:5);
numVals = size(yVec);

%%
%Part A, calculates the entropy
probClass1 = size(find(yVec==1))/numVals;
entropy = getEntropy(probClass1);

%Part B, calculates the information gain for each variable
information = zeros(1,5);
for xCol = 1:5
    xData = xVals(:,xCol);
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

[gain,colNum0] = max(information);

%information gain for all variables
information

%feature to split on initially
colNum0

%%

%Part C, calculates the decision tree

%This does the first split
[newXDataClass0, newXDataClass1,newYvecClass0, newYvecClass1...
    ,maxInfoGain,colNum ] = getDecTreeSplit( xVals,yVec );

%{
    In the first split, we use x_2
    When x_2 = 1, y=-1 so we don't need to traverse furthur down that node
    We will now split x_2=0
%}

[newXData2Class0, newXData2Class1,newYvec2Class0, newYvec2Class1...
    ,maxInfoGain2,colNum2 ] = getDecTreeSplit( newXDataClass0,newYvecClass0 );

%{
  This tells us to split on x_1
  When x_2=0,x_1=1, it happens that y=1 so we do not traverse furthur down
    We will now split x_2=0,x_1=0
%}

[newXData3Class0, newXData3Class1,newYvec3Class0, newYvec3Class1...
    ,maxInfoGain3,colNum3 ] = getDecTreeSplit( newXData2Class0,newYvec2Class0 );

%{
    This tells us to split on x_4
    When x_2=0,x_1=0,x_4=0 y=1
    When x_2=0,x_1=0,x_4=1 y=-1
    
    We are now done
%}