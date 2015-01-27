function obj = train(obj, X, Y, varargin)
% obj = train(obj, Xtrain, Ytrain [, option,val, ...])  : train logistic classifier
%     Xtrain = [n x d] training data features (constant feature not included)
%     Ytrain = [n x 1] training data classes 
%     'stepsize', val  => step size for gradient descent [default 1]
%     'stopTol',  val  => tolerance for stopping criterion [0.0]
%     'stopIter', val  => maximum number of iterations through data before stopping [1000]
%     'reg', val       => L2 regularization value [0.0]
%     'init', method   => 0: init to all zeros;  1: init to random weights;  
% Output:
%   obj.wts = [1 x d+1] vector of weights; wts(1) + wts(2)*X(:,1) + wts(3)*X(:,2) + ...


  [n,d] = size(X);            % d = dimension of data; n = number of training data

  % default options:
  plotFlag = true; 
  init     = []; 
  stopIter = 1000;
  stopTol  = -1;
  reg      = 0.0;
  stepsize = 1;

  i=1;                                       % parse through various options
  while (i<=length(varargin)),
    switch(lower(varargin{i}))
    case 'plot',      plotFlag = varargin{i+1}; i=i+1;   % plots on (true/false)
    case 'init',      init     = varargin{i+1}; i=i+1;   % init method
    case 'stopiter',  stopIter = varargin{i+1}; i=i+1;   % max # of iterations
    case 'stoptol',   stopTol  = varargin{i+1}; i=i+1;   % stopping tolerance on surrogate loss
    case 'reg',       reg      = varargin{i+1}; i=i+1;   % L2 regularization
    case 'stepsize',  stepsize = varargin{i+1}; i=i+1;   % initial stepsize
    end;
    i=i+1;
  end;

  X1    = [ones(n,1), X];     % make a version of training data with the constant feature

  Yin = Y;                              % save original Y in case needed later
  obj.classes = unique(Yin);
  if (length(obj.classes) ~= 2) error('This logistic classifier requires a binary classification problem.'); end;
  Y(Yin==obj.classes(1)) = 0;
  Y(Yin==obj.classes(2)) = 1;           % convert to classic binary labels (0/1)

  if (~isempty(init) || isempty(obj.wts))   % initialize weights and check for correct size
    obj.wts = randn(1,d+1);
  end;
  if (any( size(obj.wts) ~= [1 d+1]) ) error('Weights are not sized correctly for these data'); end;
  wtsold = 0*obj.wts+inf;

% Training loop (SGD):
iter=1; Jsur=zeros(1,stopIter); J01=zeros(1,stopIter); done=0; 
while (~done) 
  step = stepsize/iter;               % update step-size and evaluate current loss values
  
  %compute surrogate loss
  
  %computes the function given in 1d on each data point and adds it to the 
  %     total loss for this iteration
  for k = 1:length(Y)
        zValueK = dot(obj.wts,X1(k,:));
        sigmaZk = 1/(1+exp(-zValueK));
        Jsur(iter) = Jsur(iter) + -Y(k)*log(sigmaZk) + (1-Y(k))*log(1-sigmaZk) ...
            + reg*sum((obj.wts).^2);
  end
  
  %divides by number of data points to get average loss
  %     this gives us the final surrogate loss for this iteration
  Jsur(iter) = Jsur(iter)/length(Y);
  
  J01(iter) = err(obj,X,Yin);

  if (plotFlag), switch d,            % Plots to help with visualization
    case 1, fig(2); plot1DLinear(obj,X,Yin);  %  for 1D data we can display the data and the function
    case 2, fig(2); plot2DLinear(obj,X,Yin);  %  for 2D data, just the data and decision boundary
    otherwise, % no plot for higher dimensions... %  higher dimensions visualization is hard
  end; end;
  fig(1); semilogx(1:iter, Jsur(1:iter),'b-',1:iter,J01(1:iter),'g-'); drawnow;

  for j=1:n,
    
      %gets the linear response of the data point with the current weights
    zValue = dot(obj.wts,X1(j,:));

    %calculate J' vector using formula derived for it in part 1d
    sigmaZ = 1/(1+exp(-zValue));
    grad = zeros(1,length(obj.wts));
    for i = 1:length(obj.wts)
        grad(i) = X1(j,i) * (sigmaZ - Y(j)) + 2*obj.wts(i)*reg;
    end

    obj.wts = obj.wts - step * grad;      % take a step down the gradient
  end;

  done = false;
  
  %{
  Here if either the stop iteration criteria is met or the stop tolerance
        criteria is met, then this loops stops
  %}

  %sees if number of iterations is equal to stopIter
  if(iter >= stopIter)
     done = true; 
  end
  
  if(iter > 1)
      %sees if change in error is less than stopTol
     if(abs(Jsur(iter-1)-Jsur(iter))<stopTol)
        done = true; 
     end
  end
  
  wtsold = obj.wts;
  iter = iter + 1;
end;


