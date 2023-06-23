function [Z, B] = MNMST(W1, W2, lambda, d, gamma)
% ---------------------------------- data pre-processing
n = size(W1, 2);   % number of samples
% ----------------------------------------------initialize variables
B = eye(n,d);

Y1 = zeros(d,n);
Y2 = zeros(d,n);
Y3 = zeros(n,n);

H = B';
E = zeros(d,n);
Z = zeros(n,n);
J = Z;

% ----------------------------------------------
IsConverge = 0;
mu = 0.1;

w1 = 1;
w2 = 1;
W1 = SPPMI(W1, 1);
W2 = SPPMI(W2, 1);
pho = 1.5;
max_mu = 1e4;
max_iter = 100;
iter = 0;
thresh = 5e-3;
% ----------------------------------------------
while (IsConverge == 0 && iter < max_iter)
    F1 = (B' * B) \ (B' * W1);
    F2 = (B' * B) \ (B' * W2);
    F1 = max(F1,0);
    F2 = max(F2,0);
    % ---------------------------------- update B
    B = (w1 * W1 * F1' + w2 * W2 * F2' +Y2' + mu * H') / (w1 * (F1 * F1') + w2 * (F2 * F2') + mu * eye(d));
    B = max(0, B);
    % ---------------------------------- update H
    H = ((E - Y1 / mu) * (eye(n) - Z)' + B' - Y2 / mu) / ((eye(n) - Z) * (eye(n) - Z)' + eye(n));    
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
    Y2 = Y2 + mu * (H - B');
    Y3 = Y3 + mu * (Z - J + diag(diag(J)));    
    mu = min(pho*mu, max_mu);
    
    % ----------------------------------- convergence conditions    
    err1 = norm(H - H * Z - E, inf);
    err2 = norm(H - B', inf);
    err3 = norm(Z - J + diag(diag(J)), inf);
    max_err = max([err1, err2, err3]);
    
    total_loss = err1 + err2 + err3;

    if max_err < thresh
        IsConverge = 1;
    end
    disp(['Epoch is: ' num2str(iter) ', loss=' num2str(total_loss) ', self exp loss=' num2str(err1)]);
    iter = iter + 1;
end
% 使Z这个自表示学习到的矩阵对称
Z = (abs(Z) + abs(Z')) / 2;
end