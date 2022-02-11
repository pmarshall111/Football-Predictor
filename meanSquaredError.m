function error = meanSquaredError(predictions, actual) 
    numbElements = size(predictions,1) * size(predictions,2);
  
    diff = predictions-actual;
    squaredDiff = diff .^ 2;
    error = sum(sum(squaredDiff))/numbElements;
  
endfunction
