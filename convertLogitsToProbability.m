function matrix = convertLogitsToProbability(logits) 
  
  a = exp(logits);
  matrix = a ./ (1 + a);
  
  
  % normalising each row so that all the columns add up to 1.
  totalEachRow = sum(matrix, 2);
  matrix = matrix ./ totalEachRow;
  
  
endfunction
