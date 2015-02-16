function [ entropy ] = getEntropy( probClass1)
%GETENTROPY Summary of this function goes here
%   Detailed explanation goes here

probClass0 = 1-probClass1;
entropy = probClass1*log(probClass1) + probClass0*log(probClass0);
entropy = -1*entropy/log(2);

end

