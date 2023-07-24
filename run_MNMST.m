clear all;
clc;
% X = csvread("D:/st_projects/Banksy_py/data_for_matlab/multi_sec_arg.csv", 1, 1)';
% weights_adj = csvread('D:/st_projects/data_concat/concat_adj.csv', 1, 1);
% tic;
% gt = csvread('matlab_rs\concat_adata_labels.csv');

X = csvread("D:/st_projects/Banksy_py/data_for_matlab/151675_arg_data.csv", 1, 1)';
weights_adj = csvread('D:/st_projects/Banksy_py/data_for_matlab/151675_adj.csv', 1, 1);

real_label = csvread('D:/st_projects/Banksy_py/data_for_matlab/151675_real_label.csv', 1, 1);
real_label = real_label + 1;
cls_num = max(unique(real_label));
% weights_adj = data{1};

options = [];
option.Metric = 'Cosine';
options.NeighborMode = 'KNN';
options.k = 6;
options.WeightMode = 'Cosine';
cos_init = constructW(X',options);
clear options;


grid_self_results = zeros(36, 4);
grid_self_index = 0;

best_graph = zeros(size(cos_init));
best_ari = 0;
for i = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    for j = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
        grid_self_index = grid_self_index + 1;
        grid_C = self_rep(cos_init, X, i, j);
        self_grps = SpectralClustering(grid_C, cls_num);
        self_result = ClusteringMeasure_new(real_label, self_grps);
        grid_self_results(grid_self_index, :) = self_result;
        if best_ari < self_result(:, 3)
            best_ari = self_result(:, 3);
            best_graph = grid_C;
            disp(['best ari change to ', num2str(best_ari)]);
        end
        disp(['epoch is: ' num2str(grid_self_index) ', result is: ' num2str(self_result)]);
    end
end

lambda_list = [-3 -2 -1 0 1 2];
gamma_list = [-3 -2 -1 0 1 2];
% dims每36个数值更改一次
test_ari = grid_self_results(1:36, 3);
shaped_test_ari = reshape(test_ari, 6, 6);
params_fig = figure('Name', '151675');
bar3(shaped_test_ari, 0.8);
hold on;
set(gca,'xticklabel',lambda_list);
set(gca,'yticklabel', gamma_list);
xlabel('Parameter \alpha');
ylabel('Parameter \beta');
zlabel('ARI');
set(gca,'XGrid', 'on', 'YGrid', 'on','ZGrid', 'on');
% print('C:\Users\bbchond\Desktop\paper_images\grid_search\151675.eps', '-depsc', '-painters', '-r300');
% writematrix(grid_self_results, 'C:\Users\bbchond\Desktop\paper_images\grid_search\151675_alpha_beta.csv')

grid_results = zeros(36, 4);
grid_index = 0;
for i = [150]
    for j = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2] % \lambda
        for k = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2] % \gamma
            grid_index = grid_index + 1;
            [grid_Z, B, F1, F2] = MNMST(best_graph, weights_adj, j, i, k);
            grid_grps = SpectralClustering(grid_Z,cls_num);
            grid_result = ClusteringMeasure_new(real_label, grid_grps);
            grid_result;
            grid_results(grid_index, :) = grid_result;
            disp(['epoch is: ' num2str(grid_index) ', result is: ' num2str(grid_result)]);
        end
    end
end

lambda_list = [1e-3 1e-2 1e-1 1 1e1 1e2];
gamma_list = [1e-3 1e-2 1e-1 1 1e1 1e2];
test_ari = grid_results(1:36, 3);
shaped_test_ari = reshape(test_ari, 6, 6);

params_fig = figure('Name', '151675');
bar3(shaped_test_ari, 0.8);
hold on;
set(gca,'xticklabel',lambda_list);  
set(gca,'yticklabel', gamma_list);
xlabel('Parameter \lambda');
ylabel('Parameter \gamma');
zlabel('ARI');
set(gca,'XGrid', 'on', 'YGrid', 'on','ZGrid', 'on');