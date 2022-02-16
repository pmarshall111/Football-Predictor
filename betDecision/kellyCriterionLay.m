function [totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = kellyCriterionLay(betProbs, ourProbs, y, highestBy, betterThanBookiesBy)
# https://en.wikipedia.org/wiki/Kelly_criterion#Optimal_betting_example
  numbBets = 0;
  totalSpent = 0;
  totalReturn = 0;
  
  resultsToBetOn = [];
  betMatrix = [0,0; 0,0; 0,0];

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
    if lowestProb + highestBy <= secondLowest && lowestProb - betterThanBookiesBy >= bookieProb
      # do the kellyCriterion
      MAX_BET = 50;
      gainIfWin = 1/bookieProb - 1;
      stake = MAX_BET * (1-lowestProb - (lowestProb*gainIfWin))/gainIfWin;
      if stake < 0
        stake = -stake;
        liability = stake * 1/bookieProb;
        numbBets = numbBets + 1;
      
        if y(row) == lowestIndex
          totalSpent = totalSpent + liability;
          betMatrix(lowestIndex, 1) = betMatrix(lowestIndex, 1) + 1;
          resultsToBetOn = [resultsToBetOn; lowestIndex, stake, 1/bookieProb, -liability];
        else
          totalSpent = totalSpent + stake;
          totalReturn = totalReturn + stake*2;
          betMatrix(lowestIndex, 2) = betMatrix(lowestIndex, 2) + 1;
          resultsToBetOn = [resultsToBetOn; lowestIndex, stake, 1/bookieProb, stake];
        endif
      else
        resultsToBetOn = [resultsToBetOn; -1, 0, 0, 0];
      endif  
    else
      resultsToBetOn = [resultsToBetOn; -1, 0, 0, 0];
    endif
    
  endfor
  
  profit = totalReturn - totalSpent;
  percentageProfit = 100*profit / totalSpent;
