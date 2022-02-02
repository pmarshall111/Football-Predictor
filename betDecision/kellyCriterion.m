function [totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = kellyCriterion(betProbs, ourProbs, y, highestBy, betterThanBookiesBy)
# https://en.wikipedia.org/wiki/Kelly_criterion#Optimal_betting_example
  numbBets = 0;
  totalSpent = 0;
  totalReturn = 0;
  
  resultsToBetOn = [];
  betMatrix = [0,0; 0,0; 0,0];

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
    if highestProb - highestBy >= secondHighest && highestProb - betterThanBookiesBy >= bookieProb
      # do the kellyCriterion
      MAX_BET = 50;
      gainIfWin = 1/bookieProb - 1;
      stake = MAX_BET * (highestProb + (highestProb-1)/gainIfWin);
      if stake > 0
        totalSpent = totalSpent + stake;
        numbBets = numbBets + 1;
        resultsToBetOn = [resultsToBetOn; highestIndex, stake, 1/bookieProb];
      
        if y(row) == highestIndex
          totalReturn = totalReturn + stake*(1/bookieProb);
          betMatrix(highestIndex, 1) = betMatrix(highestIndex, 1) + 1;
        else
          betMatrix(highestIndex, 2) = betMatrix(highestIndex, 2) + 1;
        endif
      else
        resultsToBetOn = [resultsToBetOn; -1, 0, 0];
      endif
    else
            resultsToBetOn = [resultsToBetOn; -1, 0, 0];
    endif
    
  endfor
  
  profit = totalReturn - totalSpent;
  percentageProfit = 100*profit / totalSpent;
