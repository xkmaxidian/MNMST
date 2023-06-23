

% -------------------------------------------------------------------------
function Q = ComputeSaprseValues(H,mu)

% Q = sign(H).*max(0,abs(H- mu) ); 
% Q = Q - diag(diag(Q));


% % ----------------------------------------
% Q = max(0,H - mu) + min(0,H + mu);
% Q = Q - diag(diag(Q));
% 

% ----------------------------------------
[row, col] = size(H);
for i = 1:row
    for j=1:col
        
        if H(i, j)>mu
            Q(i, j) = H(i, j) - mu;
        elseif H(i, j) < -mu
            Q(i, j) = H(i, j) + mu;
        else
            Q(i, j) = 0;
        end
        
    end
end

Q = Q - diag(diag(Q));  % revised 0n 2017.10.16; previous codes ignore it.


% C2(N+1:N+D,:)  = max(0,(abs(Z(N+1:N+D,:)+Lambda2(N+1:N+D,:) /mu2) - 1/mu2* ones(D,N))) .* sign(Z(N+1:N+D,:)+Lambda2(N+1:N+D,:) /mu2); 
% 
% 
% Q = max(0,H)
    
    


