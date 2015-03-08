%%
X = load('data/faces.txt'); % load face dataset
i=5;
img = reshape(X(i,:),[24 24]); % convert vectorized datum to 24x24 image patch
imagesc(img); axis square; colormap gray; % display an image patch; you may have to squint

%%
%Part A
mu = mean(X,2);
muMat = repmat(mu,1,576);
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

%%

%%
%Part D
j=3;
alpha = 250;
img1 = reshape(mu(j) + alpha*V(j,:)', [24 24]);
img2 = reshape(mu(j) - alpha*V(j,:)', [24 24]);

figure
imagesc(img1); axis square; colormap gray;

figure
imagesc(img2); axis square; colormap gray;

