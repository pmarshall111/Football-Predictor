function matrix = convertVectorToMatrixWhereValIsIndex(vector) 
  
  m=numel(vector'); # converting vector y to matrix of 1s and 0s.
  n=max(vector');
  idx=sub2ind([n m],vector',1:m);
  matrix=zeros(n,m);
  matrix(idx)=1;
  matrix = matrix';
  
  
endfunction
