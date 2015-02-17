function [ entropy ] = getEntropy( probClass1)
%GETENTROPY Summary of this function goes here
%   Detailed explanation goes here

probClass0 = 1-probClass1;
entropy = 0;
if probClass1 > 0
   entropy = entropy + probClass1*log(probClass1); 
end
if probClass0 > 0
   entropy = entropy + probClass0*log(probClass0); 
end
entropy = -1*entropy/log(2);

end

