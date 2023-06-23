clear all;
clc;
X = csvread("data_for_matlab/151676_arg_data.csv", 1, 1)';
weights_adj = csvread('data_for_matlab/151676_adj.csv', 1, 1);

options = [];
option.Metric = 'Cosine';
options.NeighborMode = 'KNN';
options.k = 15;
options.WeightMode = 'Cosine';
cos_init = constructW(X',options);
W1 = SPPMI(cos_init, 1);
clear options;


cls_num = 7;
real_label = csvread('data_for_matlab/151676_real_label.csv', 1, 1);
real_label = real_label + 1;

alpha = 0.6;
beta = 0.5;
epoch = 0;
max_epoch = 200;
self_W = cos_init;
self_converge = 0;
mu = 50;
n = size(X, 2);
T1 = zeros(n, n);
T2 = zeros(n, n);
C = W1;
J1 = C;
J2 = C;

D = diag(sum(W1, 1));
L = D - W1;

while epoch < max_epoch && self_converge == 0
    % C = C .* ((X' * X + mu * (J1 - diag(diag(J1)) + J2) - T1 - T2) ./ (X' * X * C + 2 * mu * C));
    C = C .* ((X' * X + mu * (J1 - diag(diag(J1))) - T1 + beta * W1 * C) ./ (X' * X * C + mu * C + beta * D * C));
    C = C - diag(diag(C));
    J1 = soft(C + T1 / mu, alpha / mu);
    J1 = J1 - diag(diag(J1));
    % J2 = softth(C + T2 / mu, beta / mu);
    %%%%%===========Update Lagrange multiplier T ===========
    T1 = T1 + mu * (C - J1);
    % T2 = T2 + mu * (C - J2);
    %%%%%===========X - XC ===========
    err = norm(X - X * C, 'fro');
    if err < 1e-2
        self_converge = 1;
    end
    disp(['Epoch is: ' num2str(epoch) ', loss=' num2str(err)]);
    epoch = epoch + 1;
end
C = (abs(C) + abs(C')) / 2;
self_grps = SpectralClustering(C, cls_num);
self_result = ClusteringMeasure_new(real_label, self_grps);

init_grps = SpectralClustering(W1, cls_num);
init_result = ClusteringMeasure_new(real_label, init_grps);
disp(init_result);
disp(self_result);  

%------- 可调的几个参数
d = 200;
lambda = 10;
gamma = 1;
[Z, B] = MNMST(C, weights_adj, lambda, d, gamma);
grps = SpectralClustering(Z,cls_num);
% ACC NMI ARI F-score
result = ClusteringMeasure_new(real_label, grps);
writematrix(B, 'matlab_rs/learned_151675_common_B.csv')
writematrix(Z, 'matlab_rs/learned_151675.csv')

% grid_results = zeros(147, 4);
% grid_index = 0;
% for i = [100, 150, 200]
%     for j = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
%         for k = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
%             grid_index = grid_index + 1;
%             [grid_Z] = multi_view(SPPMI(C, 1), W2, j, i, k);
%             grid_grps = SpectralClustering(grid_Z,cls_num);
%             grid_result = ClusteringMeasure_new(real_label, grid_grps);
%             grid_results(grid_index, :) = grid_result;
%             disp(['epoch is: ' num2str(grid_index) ', result is: ' grid_result]);
%         end
%     end
% end
% writematrix(grid_results, 'grid_search_result_151509.csv')


function[y] = soft( x, T )
    if sum( abs(T(:)) )==0
        y = x;
    else
        y = max( abs(x) - T, 0);
        y = sign(x).*y;
    end
end