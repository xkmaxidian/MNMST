clear all;
clc;
% X = csvread("D:/st_projects/data/data_for_matlab/151676_arg_data.csv", 1, 1)';
weights_adj = csvread('D:/st_projects/data/data_for_matlab/151675_adj.csv', 1, 1);
X = csvread('D:/st_projects/data/data_for_matlab/151675_arg_data.csv', 1, 1)';
% image = csvread('D:/st_projects/data/data_for_matlab/151676_image_pca.csv', 1, 1)';

options = [];
option.Metric = 'Cosine';
options.NeighborMode = 'KNN';
options.k = 6;
options.WeightMode = 'Cosine';
cos_init = constructW(X',options);


W1 = SPPMI(cos_init, 1);
W2 = SPPMI(weights_adj, 1);
clear options;


cls_num = 7;
real_label = csvread('D:/st_projects/data/data_for_matlab/151675_real_label.csv', 1, 1);
real_label = real_label + 1;

% grid_self_results = zeros(49, 4);
% grid_self_index = 0;
% 
% for i = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
%     for j = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
%         grid_self_index = grid_self_index + 1;
%         C = self_rep(cos_init, X, i, j);
%         self_grps = SpectralClustering(C, cls_num);
%         self_result = ClusteringMeasure_new(real_label, self_grps);
%         grid_self_results(grid_self_index, :) = self_result;
%         disp(['epoch is: ' num2str(grid_self_index) ', result is: ' num2str(self_result)]);
%     end
% end

% image_C = self_rep(image_init, image, 1, 0.01);

% self_grps = SpectralClustering(C, cls_num);
% self_result = ClusteringMeasure_new(real_label, self_grps);
% disp(self_result);
% init_grps = SpectralClustering(W1, cls_num);
% init_result = ClusteringMeasure_new(real_label, init_grps);
% disp(init_result)
% image_self_grps = SpectralClustering(image_C, cls_num);
% image_self_result = ClusteringMeasure_new(real_label, image_self_grps);
% image_init_grps = SpectralClustering(image_init, cls_num);
% image_init_result = ClusteringMeasure_new(real_label, image_init_grps);
% disp(image_init_result)
tic;
C= self_rep(cos_init, X, 1, 1);
%------- 可调的几个参数
d = 150;
lambda = 10;
gamma = 10;
[Z, B, F1, F2] = MNMST(C, weights_adj, lambda, d, gamma);
% show executing time
elapsedTime = toc;
disp(['Elapsed Time: ' num2str(elapsedTime / 60) ' mins']);
mem = memory;
% display memory useage
disp(['Memory Used: ' num2str(mem.MemUsedMATLAB/1024/1024 / 1024) 'GB']);


grps = SpectralClustering(Z,cls_num);
% ACC NMI ARI F-score
result = ClusteringMeasure_new(real_label, grps);
disp(result);

% writematrix(B, 'matlab_rs/learned_151669_common_B.csv')
%  writematrix(Z, 'matlab_rs/learned_151669.csv')

grid_results = zeros(196, 4);
grid_index = 0;
for i = [50, 100, 150, 200]
    for j = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        for k = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
            grid_index = grid_index + 1;
            [grid_Z, B, F1, F2] = MNMST(cos_init, image_init, j, i, k);
            grid_grps = SpectralClustering(grid_Z,cls_num);
            grid_result = ClusteringMeasure_new(real_label, grid_grps);
        end
    end
end


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
