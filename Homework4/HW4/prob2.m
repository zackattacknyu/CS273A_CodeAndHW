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

probClass1 = size(find(yVec==1))/size(yVec);
probClassMinus1 = 1-probClass1;

entropy = probClass1*log(probClass1) + probClassMinus1*log(probClassMinus1);
entropy = -1*entropy/log(2);