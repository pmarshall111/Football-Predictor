function regProbs = regProbabilities(probs)
  totalEachRow = sum(probs, 2);
  regProbs = probs ./ totalEachRow;
endfunction
