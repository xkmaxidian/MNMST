% 此函数采用一个图的邻接矩阵并使用谱聚类算法计算节点的聚类情况
% CMat: NxN的邻接矩阵
% n: 聚类的簇数
% groups: N维向量，包含通过谱聚类获得的n个簇的对应成员



function groups = SpectralClustering(CKSym,n)

warning off;
N = size(CKSym,1);

% KMeans的最大迭代次数
MAXiter = 1000;

% KMeans的复制数量
REPlic = 20;

% 归一化谱聚类
% 使用归一化对称拉普拉斯矩阵 L = I - D^{-1/2} W D^{-1/2}

DN = diag( 1./sqrt(sum(CKSym)+eps));
LapN = double(speye(N)) - double(DN * CKSym * DN);   %%%  double
[uN,sN,vN] = svd(LapN);
kerN = vN(:,N-n+1:N);
for i = 1:N
    kerNS(i,:) = kerN(i,:) ./ norm(kerN(i,:)+eps);
end

%使用KMeans进行聚类
groups = kmeans(kerNS,n,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');