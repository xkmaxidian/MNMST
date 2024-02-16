function [Z] = gene_co_exp(X, threshold)
    correlationMatrix = corr(X);
    correlationMatrix(abs(correlationMatrix) < threshold) = 0;
    Z = correlationMatrix;
    Z = (Z + Z') / 2;
end