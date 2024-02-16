clear all;
clc;
X = csvread("D:/st_projects/Banksy_py/data_for_matlab/151510_arg_data.csv", 1, 1)';
weights_adj = csvread('D:/st_projects/Banksy_py/data_for_matlab/151510_adj.csv', 1, 1);

cls_num = 7;
real_label = csvread('D:/st_projects/Banksy_py/data_for_matlab/151510_real_label.csv', 1, 1);
real_label = real_label + 1;

K = 15;
co_exp_network = gene_co_exp(X, 0);
matrix_topK = zeros(size(co_exp_network));
for i = 1:size(co_exp_network, 1)
    [~, indices] = sort(co_exp_network(i, :), 'descend');
    matrix_topK(i, indices(1:K)) = co_exp_network(i, indices(1:K));
end
init_grps = SpectralClustering(matrix_topK, cls_num);
init_result = ClusteringMeasure_new(real_label, init_grps);
disp(init_result);

d = 150;
lambda = 10;
gamma = 1;
[Z, B, F1, F2] = MNMST(matrix_topK, weights_adj, lambda, d, gamma);
% toc
grps = SpectralClustering(Z, cls_num);
% ACC NMI ARI F-score
result = ClusteringMeasure_new(real_label, grps);
disp(result);
writematrix(Z, 'matlab_rs/Pearson_affinity_151510.csv')
