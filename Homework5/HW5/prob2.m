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

%Part A and B
K=20;
numTrials = 4;
scores = zeros(1,numTrials);
bestScore = Inf;
for j=1:numTrials
    [zCur,cCur,score] = kmeans(Xn,K);
    scores(j) = score;
    if(score < bestScore)
       z = zCur;
       c = cCur;
       bestScore = score;
    end
end

%%

%Part C

%gets the number of documents per cluster
[numDocsPerCluster,Clusters] = hist(z,unique(z));

%prints out the clusters
for i=1:K
   [~,orderI] = sort( c(i,:), 'descend');
   fprintf('Cluster %d: ',i); fprintf('%s ',vocab{orderI(1:10)}); fprintf('\n');
end

%%

%Part D
%gets the assignments for docs 1,15,30
assign1 = z(1);
assign15 = z(15);
assign30 = z(30);

docsWithSameAs1 = find(z==assign1);
docsWithSameAs15 = find(z==assign15);
docsWithSameAs30 = find(z==assign30);