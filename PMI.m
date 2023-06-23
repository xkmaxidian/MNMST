function [ SPPMI ] = createSPPMIMtx(G , k)
%% Creating SPPMI Matrix using graph G
% Calculating Degrees for each node
  nodeDegrees = sum(G);
  W = sum(nodeDegrees);
  SPPMI = G;
% use a loop to calculate Wij*W/(di*dj)
  [col,row,weights] = find(G);
  for i = 1:length(col)
          score = log(weights(i) * W / nodeDegrees(col(i)) / nodeDegrees(row(i))) - log(k);
          if(score > 0)
            SPPMI(col(i),row(i)) = score;
          else
              SPPMI(col(i),row(i)) = 0;
          end
  end

%   spfun(@shiftOpt,SPPMI);
%     function score = shiftOpt(x)
%         score = log(x) - log(k);
%         if(score<0)
%             score = 0;
%         end
%     end
    disp('Shifted PMI Matrix is Done');
end

