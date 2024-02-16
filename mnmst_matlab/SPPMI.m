function [ SPPMI ] = createSPPMIMtx(G , k)
%% Creating SPPMI Matrix using graph G
% Calculating Degrees for each node
  nodeDegrees = sum(G);   %每一列求和
  nodeDegrees2=sum(G,2);  %每一行求和
  W = sum(nodeDegrees);   %总权值
  SPPMI = G;
% use a loop to calculate Wij*W/(di*dj)
  [col,row,weights] = find(G);
  for i = 1:length(col)
          score = log(weights(i) * W / nodeDegrees2(col(i)) / nodeDegrees(row(i))) - log(k);
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
%     disp('SPPMI Matrix is　Done');
end

