%%
X = load('data/faces.txt'); % load face dataset
i=5;
img = reshape(X(i,:),[24 24]); % convert vectorized datum to 24x24 image patch
imagesc(img); axis square; colormap gray; % display an image patch; you may have to squint

%%
%Part A
mu = mean(X);
muMat = repmat(mu,size(X,1),1);
X_0 = X-muMat;

%%
%Part B
[U,S,V] = svd(X_0);
W = U*S;

%%
%Part C
mseSVD = zeros(1,10);
for K=1:10
   X_0hat = W(:,1:K)*(V(:,1:K)');
   mseSVD(K) = mean( mean( (X_0hat-X_0).^2 ) );
end
plot(mseSVD);
xlabel('K value');
ylabel('Mean Squared Error');
title('MSE as function of K');

%%
%Part D
j=1;
alpha = 2*median(abs(W(:,j)));
img1 = reshape(mu + alpha*V(:,j)', [24 24]);
img2 = reshape(mu - alpha*V(:,j)', [24 24]);

figure
imagesc(img1); axis square; colormap gray;

figure
imagesc(img2); axis square; colormap gray;

%%

%Part E
idx = 20:30; % pick some data at random or otherwise
figure; hold on; axis ij; colormap(gray);
range = max(W(idx,1:2)) - min(W(idx,1:2)); % find range of coordinates to be plotted
scale = [200 200]./range; % want 24x24 to be visible but not large on new scale
for i=idx, imagesc(W(i,1)*scale(1),W(i,2)*scale(2), reshape(X(i,:),24,24)); end;

%%

%Part F
K=50; %need to test K=5,10,50
faceNum = 15; %test 10,15
X_0hat = W(:,1:K)*(V(:,1:K)');
imgRecon = reshape(X_0hat(faceNum,:),[24 24]);
figure
imagesc(imgRecon); axis square; colormap gray;
figure
imagesc(reshape(X(faceNum,:),[24 24])); axis square; colormap gray;


