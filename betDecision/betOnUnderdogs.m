function [totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = betOnUnderdogs(betProbs, ourProbs, y, highestBy, betterThanBookiesBy)
  
  numbBets = 0;
  totalSpent = 0;
  totalReturn = 0;
  
  betMatrix = [0,0; 0,0; 0,0];
  resultsToBetOn = [];

  for row = 1:size(ourProbs,1)
    lowestProb = 1;
    lowestIndex = -1;
    secondLowest = 1;
    secondLowestIndex = -1;
    
    for col = 1:size(ourProbs,2) # calc highest probability
      currProb = ourProbs(row,col);
      if currProb < lowestProb
        secondLowest = lowestProb;
        secondLowestIndex = lowestIndex;
        lowestProb = currProb;
        lowestIndex = col;
      elseif currProb < secondLowest
        secondLowest = currProb;
        secondLowestIndex = col;
      endif
    endfor
    
    bookieProb = betProbs(row, lowestIndex);
    if lowestProb + highestBy <= secondLowest && lowestProb - betterThanBookiesBy >= bookieProb && bookieProb != -1
      # do the variable stake
      stake = 10;
      probAboveBetterThanBookiesBy = (lowestProb - bookieProb - betterThanBookiesBy);
      currMult = 20*probAboveBetterThanBookiesBy + 1;
      stake = currMult*stake;
      totalSpent = totalSpent + stake;
      numbBets = numbBets + 1;
      resultsToBetOn = [resultsToBetOn; lowestIndex, stake, 1/bookieProb];
      
      
      # work out if we won
      if y(row) == lowestIndex
        decimalOdds = 1/bookieProb;
        totalReturn = totalReturn + stake*decimalOdds;
        betMatrix(lowestIndex, 1) = betMatrix(lowestIndex, 1) + 1;
      else
        betMatrix(lowestIndex, 2) = betMatrix(lowestIndex, 2) + 1;
      endif
            
    else
      resultsToBetOn = [resultsToBetOn; -1, 0, 0];
            
    endif
    
  endfor
  
  profit = totalReturn - totalSpent;
  
  if totalSpent == 0
    percentageProfit = 0;
  else 
    percentageProfit = 100*profit / totalSpent;
  endif
  
  #fprintf("\n\nY axis is Win/Draw/Loss. X axis is Correct/Incorrect\n")
  #betMatrix
  
endfunction
