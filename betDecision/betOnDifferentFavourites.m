function [totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = betOnDifferentFavourites(betProbs, ourProbs, y)
  
  numbBets = 0;
  totalSpent = 0;
  totalReturn = 0;
  
  betMatrix = [0,0; 0,0; 0,0];
  resultsToBetOn = [];

  stake=5;
  
  [maxVal, maxIdx] = max(betProbs,[],2);
  [ourMaxVal, ourMaxIdx] = max(ourProbs,[],2);
  
  toBetOn = ourMaxIdx != maxIdx;
  resToBetOn = toBetOn .* ourMaxIdx;
  resToBetOnMtx = [];
  for i=1:length(resToBetOn)
    resToBetOnMtx(i,:) = zeros(3,1);
    if resToBetOn(i) > 0
      resToBetOnMtx(i,resToBetOn(i)) = 1;
    endif
  endfor
  
  
  numbBets = sum(sum(resToBetOnMtx));
  bets = resToBetOnMtx .* stake;
  totalSpent = sum(sum(bets));
  
  # Creating 1s in position of correct results
  correctRes = convertVectorToMatrixWhereValIsIndex(y);
  betOdds = 1 ./ betProbs;
  totalReturn = sum(sum(betOdds .* correctRes .* bets));
  
  profit = totalReturn - totalSpent;
  
  if totalSpent == 0
    percentageProfit = 0;
  else 
    percentageProfit = 100*profit / totalSpent;
  endif
  
  #fprintf("\n\nY axis is Win/Draw/Loss. X axis is Correct/Incorrect\n")
  #betMatrix
  
endfunction
