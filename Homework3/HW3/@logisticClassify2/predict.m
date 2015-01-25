function Yte = predict(obj,Xte)
% Yhat = predict(obj, X)  : make predictions on test data X

% (1) make predictions based on the sign of wts(1) + wts(2)*x(:,1) + ...
weights = obj.wts;
yhat = weights(1);
for i = 2:length(weights)
    yhat = yhat + Xte(:,i-1).*weights(i);
end
yhat = sign(yhat);

% (2) convert predictions to saved classes: Yte = obj.classes( [1 or 2] );
Yte = ones(1,length(yhat));
for i = 1:length(Yte)
   if(yhat(i) == -1)
      Yte(i) = obj.classes(1); 
   else
       Yte(i) = obj.classes(2);
   end
end