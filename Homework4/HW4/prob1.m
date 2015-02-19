iris=load('data/iris.txt'); % load the text file
X = iris(:,1:2); Y=iris(:,end); % get first two features

% get class 0 vs 1
XA = X(Y<2,:); YA=Y(Y<2); 
sizeXA = size(YA);
numFeatures = sizeXA(1);

Yvar = YA.*2 - 1; %turns 0 and 1 into -1 and 1
%%
%{
we need to minimize sum w_i^2
subject to y(w*x+b) >= 1

The x vector for quadprog is as follows:
b
w_1
w_2

The minimizing matrix H would then be:
0 0 0
0 2 0
0 0 2

f would then be all zeros

I take the negative of the constraint to get
-y(i)*b + -y*w_1*x_1(i) + -y*w_2*x_2(i) <= -1
and that form is compatible with quadprog

each row of A for quadprog is as follows then:
-y(i) -y(i)*x_1(i) -y(i)*x_2(i)

each row of b for quadprog is just -1

%}

%H,f for quadprog as described above
H = [0 0 0; 0 2 0;0 0 2];
f = [0 0 0];

%gets the A matrix for quadprog
A = [ones(numFeatures,1) XA];
for row = 1:numFeatures
   A(row,:) = A(row,:).*(Yvar(row)*-1);
end

%gets the b column
b = ones(numFeatures,1).*-1;

theta = quadprog(H,f,A,b);

%%

b = theta(1);
w = theta(2:3);
result = abs(XA*w + b);
[newRes Inds] = sort(result);
supportVecInds = Inds(1:3);
supportVecs = XA(supportVecInds,:);

%%
%plots the classification boundary from this
learnerA=logisticClassify();
learnerA=setClasses(learnerA, unique(YA));
learnerA=setWeights(learnerA, theta');
plotClassify2D(learnerA,XA,YA);
hold on
plot(supportVecs(:,1),supportVecs(:,2),'bx');

%%

%{
We have to change the optimization function to
min alpha>=0 1/2 sum_{alpha_i...} - sum_{alpha_i}
%}

%This will let us construct the H matrix for quadprog
Kmat = XA*XA';
yMat = Yvar*Yvar';
Hmatrix = yMat.*Kmat;

%this is the f vector
fVec = -ones(numFeatures,1);

%this is the A matrix, which will also be Aeq
Amat = Yvar';
bVal = 0;

%the lower bound is zero
LBvec = zeros(numFeatures,1);

%run quadprog
alpha = quadprog(Hmatrix,fVec,[],[],Amat,bVal,LBvec);