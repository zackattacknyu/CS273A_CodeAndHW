function plot2DLinear(obj, X, Y)
% plot2DLinear(obj, X,Y)
%   plot a linear classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
%
  [n,d] = size(X);
  if (d~=2) error('Sorry -- plot2DLogistic only works on 2D data...'); end;

  weights = obj.wts;
  xs = min(X):0.05:max(X);
  ys = -(xs.*weights(2) + weights(1))/(weights(3));
  
  scatter(X(:,1),X(:,2),10,Y);
  hold on
  plot(xs,ys);
  
