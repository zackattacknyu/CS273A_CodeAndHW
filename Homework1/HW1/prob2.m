iris=load('data/iris.txt'); y=iris(:,end); X=iris(:,1:end-1);

% Note: indexing with ":" indicates all values (in this case, all rows);
% indexing with a value ("1", "end", etc.) extracts only that one value (here, columns);
% indexing rows/columns with a range ("1:end-1") extracts any row/column in that range.

[X y] = shuffleData(X,y); % shuffle data randomly
% (This is a good idea in case your data are ordered in some pathological way,
% as the Iris data are)
[Xtr Xte Ytr Yte] = splitData(X,y, .75); % split data into 75/25 train/test