% Read in vocabulary and data (word counts per document)
[vocab] = textread('data/text/vocab.txt','%s');
[did,wid,cnt] = textread('data/text/docword.txt','%d%d%d','headerlines',3);
X = sparse(did,wid,cnt); % convert to a matlab sparse matrix
D = max(did); % number of docs
W = max(wid); % size of vocab
N = sum(cnt); % total number of words
% It is often helpful to normalize by the document length:
Xn= X./repmat(sum(X,2),[1,W]) ; % divide word counts by doc length

for i=1:size(Xn,1)
   [sorted,order] = sort( Xn(i,:), 2, 'descend');
    fprintf('Doc %d: ',i); fprintf('%s ',vocab{order(1:10)}); fprintf('\n'); 
end
%%

scores = zeros(1,4);
bestScore = Inf;
for j=1:4
    [zCur,cCur,score] = kmeans(Xn,20);
    scores(j) = score;
    if(score < bestScore)
       z = zCur;
       c = cCur;
       bestScore = score;
    end
end