function [totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = BTB_VariableStake(betProbs, ourProbs, y, highestBy, betterThanBookiesBy)
  
  numbBets = 0;
  totalSpent = 0;
  totalReturn = 0;
  
  betMatrix = [0,0; 0,0; 0,0];
  resultsToBetOn = [];

  for row = 1:size(ourProbs,1)
    highestProb = 0;
    highestIndex = -1;
    secondHighest = 0;
    secondHighestIndex = -1;
    
    for col = 1:size(ourProbs,2) # calc highest probability
      currProb = ourProbs(row,col);
      if currProb > highestProb
        secondHighest = highestProb;
        secondHighestIndex = highestIndex;
        highestProb = currProb;
        highestIndex = col;
      elseif currProb > secondHighest
        secondHighest = currProb;
        secondHighestIndex = col;
      endif
    endfor
    
    bookieProb = betProbs(row, highestIndex);
    if highestProb - highestBy >= secondHighest && highestProb - betterThanBookiesBy >= bookieProb && bookieProb != -1
      # do the variable stake
      stake = 10;
      probAboveBetterThanBookiesBy = (highestProb - betProbs(row,highestIndex) - betterThanBookiesBy);
      currMult = 20*probAboveBetterThanBookiesBy + 1;
      stake = currMult*stake;
      totalSpent = totalSpent + stake;
      numbBets = numbBets + 1;
      
      
      # work out if we won
      if y(row) == highestIndex
        decimalOdds = 1/bookieProb;
        totalReturn = totalReturn + stake*decimalOdds;
        betMatrix(highestIndex, 1) = betMatrix(highestIndex, 1) + 1;
        resultsToBetOn = [resultsToBetOn; highestIndex, stake, 1/bookieProb, stake*decimalOdds-stake];
      else
        betMatrix(highestIndex, 2) = betMatrix(highestIndex, 2) + 1;
        resultsToBetOn = [resultsToBetOn; highestIndex, stake, 1/bookieProb, -stake];
      endif
      
    else
      resultsToBetOn = [resultsToBetOn; -1, 0, 0, 0];
                  
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
