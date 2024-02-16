function [C] = self_rep(init_W, X, alpha, beta)
epoch = 0;
max_epoch = 100;
self_converge = 0;
mu = 50;
n = size(X, 2);
T1 = zeros(n, n);
C = init_W;
J1 = C;

D = diag(sum(init_W, 1));

while epoch < max_epoch && self_converge == 0
    % C = C .* ((X' * X + mu * (J1 - diag(diag(J1)) + J2) - T1 - T2) ./ (X' * X * C + 2 * mu * C));
    C = C .* ((X' * X + mu * (J1 - diag(diag(J1))) - T1 + beta * init_W * C) ./ (X' * X * C + mu * C + beta * D * C));
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
end

function[y] = soft(x, T)
    if sum( abs(T(:)) )==0
        y = x;
    else
        y = max(abs(x) - T, 0);
        y = sign(x).*y;
    end
end