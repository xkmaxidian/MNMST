clear all;
clc;
X = csvread("D:/st_projects/Banksy_py/data_for_matlab/151508_arg_data.csv", 1, 1)';
weights_adj = csvread('D:/st_projects/Banksy_py/data_for_matlab/151508_adj.csv', 1, 1);

cls_num = 5;
real_label = csvread('D:/st_projects/Banksy_py/data_for_matlab/151508_real_label.csv', 1, 1);
real_label = real_label + 1;

correlationMatrix = corr(X, 'Type', 'Spearman');
correlationMatrix(abs(correlationMatrix) < 0.5) = 0;
spear_man = (correlationMatrix + correlationMatrix') / 2;

d = 150;
lambda = 10;
gamma = 1;
[Z, B, F1, F2] = MNMST(spear_man, weights_adj, lambda, d, gamma);
% toc
grps = SpectralClustering(Z,cls_num);
% ACC NMI ARI F-score
result = ClusteringMeasure_new(real_label, grps);
disp(result);
writematrix(Z, 'matlab_rs/Spearman_affinity_151508.csv')
