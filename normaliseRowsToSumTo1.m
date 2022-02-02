function output = normaliseRowsToSumTo1(input) 
  % normalising each row so that all the columns add up to 1.
  totalEachRow = sum(input, 2);
  output = input ./ totalEachRow;
  
endfunction