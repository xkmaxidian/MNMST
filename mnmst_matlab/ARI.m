function [ AR ] = ARI(Clustering1,k1,Clustering2,k2)
% This function returns Adjusted Rand Index ( Hubert & Arabie) of two clusterings 1 & 2.
%variable 'Clustering1' is Nx1 vector with an integer number between 1
%and K1 to denote which cluster the corresponding data point assigned to in
%the first clustering. Similarly for 'Clustering2'
N=size(Clustering1,1);
contig_matrix= zeros(k1,k2);

for point=1:N
   i=Clustering1(point) ;
   j=Clustering2(point);
  contig_matrix(i, j) = contig_matrix(i, j)+1;
    
end


a= sum(contig_matrix');
b=sum(contig_matrix);

SumCombnij=0;

for i=1:k1
    for j=1:k2
        
    if (contig_matrix(i,j)>1) 
        SumCombnij=SumCombnij+ nchoosek(contig_matrix(i,j),2) ;
    end  
        
        
    end 
end

SumCombai=0;
for i=1:k1
   if ( a(i)>1)
         SumCombai=  SumCombai+nchoosek(a(i),2);
   end 
    
end
SumCombbj=0;
for j=1:k2
   if ( b(j)>1)
         SumCombbj=  SumCombbj+nchoosek(b(j),2);
   end 
    
end
nCh2=nchoosek(N,2);
temp=(SumCombai*SumCombbj)/nCh2;

AR =(SumCombnij-temp)/(0.5*(SumCombai+SumCombbj)-temp);




end

