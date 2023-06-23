function [E] = solve_l1l2(W,lambda)
% n = size(W,2);
n = size(W,1);
E = W;
for i=1:n
%     E(:,i) = solve_l2(W(:,i),lambda);
    E(i,:) = solve_l2(W(i,:),lambda);
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
end