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

%Part A and B. scores(1) is the result for Part A.
K=20;
numTrials = 5;
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
%prints out the histogram of num docs per cluster
hist(z,20)
xlabel('Cluster Number');
ylabel('Number of Documents in Cluster');
title('Number of Documents per Cluster');

%%

%Part D
%gets the assignments for docs 1,15,30
docNums = [1 15 30];
assignments = zeros(1,3);
docsWithSameCluster = cell(1,3);
for i = 1:3
   assignments(i) = z(docNums(i)); 
   docsWithSameCluster{i} = find(z == assignments(i));
end

for cluster = 1:3
    currentDocs = docsWithSameCluster{cluster};
   for doc = 1:min(12,length(currentDocs));
       curDocNum = currentDocs(doc);
       fname = sprintf('data/text/example1/20000101.%04d.txt',curDocNum);
        txt = textread(fname,'%s',10,'whitespace','\r\n'); 
        fprintf('%s\n',txt{:});
        fprintf('\n');
   end
   fprintf('\n\n\n');
end
%%

%Part E
K=40;
numTrials = 4;
scores = zeros(1,numTrials);
bestScore = Inf;
for j=1:numTrials
    [zCur,cCur,score] = kmeans(Xn,K,'k++');
    scores(j) = score;
    if(score < bestScore)
       z = zCur;
       c = cCur;
       bestScore = score;
    end
end

%gets the assignments for docs 1,15,30
docNums = [1 15 30];
assignments = zeros(1,3);
docsWithSameCluster = cell(1,3);
for i = 1:3
   assignments(i) = z(docNums(i)); 
   docsWithSameCluster{i} = find(z == assignments(i));
end

for cluster = 1:3
    currentDocs = docsWithSameCluster{cluster};
   for doc = 1:min(12,length(currentDocs));
       curDocNum = currentDocs(doc);
       fname = sprintf('data/text/example1/20000101.%04d.txt',curDocNum);
        txt = textread(fname,'%s',10,'whitespace','\r\n'); 
        fprintf('%s\n',txt{:});
        fprintf('\n');
   end
   fprintf('\n\n\n');
end