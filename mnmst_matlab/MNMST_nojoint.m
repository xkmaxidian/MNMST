function [Z, B, F1, F2] = MNMST_nojoint(W1, W2, lambda, d, gamma)
% ---------------------------------- data pre-processing
n = size(W1, 2);   % number of samples
% ----------------------------------------------initialize variables
B = eye(n,d);

Y1 = zeros(d,n);
Y3 = zeros(n,n);
E = zeros(d,n);
Z = zeros(n,n);
J = Z;

% ----------------------------------------------
IsConverge = 0;
mu = 0.1;

w1 = 1;
w2 = 1;
W1 = SPPMI(W1, 1);
% W2 = SPPMI(W2, 1);
pho = 1.5;
max_mu = 1e4;
max_iter = 100;
iter = 0;
thresh = 1e-2;

% ----------------------------------------------
while (IsConverge == 0 && iter < max_iter)
    F1 = (B' * B) \ (B' * W1);
    F2 = (B' * B) \ (B' * W2);
    F1 = max(F1,0);
    F2 = max(F2,0);
    % ---------------------------------- update B
    B = (w1 * W1 * F1' + w2 * W2 * F2') / (w1 * (F1 * F1') + w2 * (F2 * F2'));
    B = max(0, B);
    err1 = norm(W1 - B * F1, inf);
    err2 = norm(W2 - B * F2, inf);
    max_err = max([err1, err2]);
    if max_err < thresh
        IsConverge = 1;
    end
    iter = iter + 1;
end
iter = 0;

IsConverge = 0;
H = B';
while (iter < max_iter && IsConverge == 0)
    % ---------------------------------- update Z
    Z = (H' * H + eye(n)) \ (H' * (H - E) + (H' * Y1 - Y3) / mu + (J-diag(diag(J))));
    % ---------------------------------- update E
    A = H - (H * Z) + (Y1 / mu);
    % 参数lambda是要手动输入的可调参数
    E = solve_l1l2(A, lambda / mu);
    % ---------------------------------- update J
    % Nuclear norm， 参数gamma是要手动输入的可调参数
    J = softth(Z + Y3/mu, gamma/mu);    
    % ---------------------------------- updata multipliers
    
    Y1 = Y1 + mu * (H - H * Z - E);
    Y3 = Y3 + mu * (Z - J + diag(diag(J)));    
    mu = min(pho*mu, max_mu);
    
    % ----------------------------------- convergence conditions    
    err1 = norm(H - H * Z - E, inf);
    err3 = norm(Z - J + diag(diag(J)), inf);
    max_err = max([err1, err3]);
    
    total_loss = err1 + err3;

    if max_err < thresh
        IsConverge = 1;
    end
    disp(['Epoch is: ' num2str(iter) ', loss=' num2str(total_loss) ', self exp loss=' num2str(err1)]);
    iter = iter + 1;
end
% 使Z这个自表示学习到的矩阵对称
Z = (abs(Z) + abs(Z')) / 2;
end