function score = brierScore(probs,y)
  score = 0;
  
  for row=1:length(y)
    for col = 1:size(probs,2)
      currProb = probs(row,col);
      ans = y(row);
      if ans == col
        score += (currProb-1)^2;
      else
        score += (currProb-0)^2;
      endif
    endfor
  endfor
  
  score = score / (size(probs,1)*size(probs,2));
endfunction
