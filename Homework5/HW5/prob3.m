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
