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
Ainit = [ones(numFeatures,1) XA];
A = -1.*Ainit.*(repmat(Yvar,1,3));

%gets the b column
b = ones(numFeatures,1).*-1;

theta = quadprog(H,f,A,b);

%%

%{
We have to change the optimization function to
min alpha>=0 1/2 sum{alpha_i alpha_j y(i) y(j) K_ij} - sum alpha_i

Details below on what the variables become in order to 
    compute the quadratic program in this form
%}

%this lets us construct the dot product matrix K
Kmat = XA*XA';

%this constructs the matrix H for the program
yMat = Yvar*Yvar';
Hmatrix = yMat.*Kmat;

%this is the f vector
fVec = -ones(numFeatures,1);

%this is the A matrix, which will also be Aeq
Amat = Yvar';

%this is the b value for the input
bVal = 0;

%the lower bound is zero
LBvec = zeros(numFeatures,1);

%run quadprog on dual form finally
alpha = quadprog(Hmatrix,fVec,[],[],Amat,bVal,LBvec);

%%

%this part verifies the alpha solution and then plots the 
%       boundary and support vectors

%gets b,w from theta
bFromTheta = theta(1);
wFromTheta = theta(2:3);

%due to floating part error, this is just to indicate almost zero
epsilon = 0.001;

%I find the support vectors here
%the alphas that are greater than 0 indicate support vectors
supportVecInds = find(alpha>epsilon);
supportVecs = XA(supportVecInds,:);
supportVecsY = Yvar(supportVecInds);

%verify w obtained from alpha
wFromAlpha = (alpha.*Yvar)'*XA;
diffBetweenW = sum(abs(wFromAlpha'-wFromTheta));
assert(diffBetweenW < epsilon); %no assertion failed occured, so this is true

%verify b obtained from alpha
bFromAlpha = mean(supportVecsY-supportVecs*wFromAlpha');
diffBetweenB = abs(bFromAlpha-bFromTheta);
assert(diffBetweenB < epsilon); %no assertion fail

%plots the classification boundary finally
learnerA=logisticClassify();
learnerA=setClasses(learnerA, unique(YA));
learnerA=setWeights(learnerA, theta');
plotClassify2D(learnerA,XA,YA);
hold on
plot(supportVecs(:,1),supportVecs(:,2),'y*');