function [totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = betOnCloseGamesBTB(betProbs, ourProbs, y, betterThanBookiesBy)
  
  numbBets = 0;
  totalSpent = 0;
  totalReturn = 0;
  
  
  betMatrix = [0,0; 0,0; 0,0];
  resultsToBetOn = [];

  stake=5;
  toBetOn = ourProbs+betterThanBookiesBy < betProbs;
  
  closeGames = abs(betProbs(:,1:1) - betProbs(:,3:3)) < 0.25;
  toBetOn = toBetOn .* closeGames;
  
  numbBets = sum(sum(toBetOn));
  bets = toBetOn .* stake;
  totalSpent = sum(sum(bets));
 
  
  # Creating 1s in position of correct results
m=numel(y');
n=max(y');
idx=sub2ind([n m],y',1:m);
correctRes=zeros(n,m);
correctRes(idx)=1;
correctRes = correctRes';
  
  totalReturn = sum(sum((1 ./betProbs) .*correctRes .* bets));
  
  profit = totalReturn - totalSpent;
  
  if totalSpent == 0
    percentageProfit = 0;
  else 
    percentageProfit = 100*profit / totalSpent;
  endif
  
  #fprintf("\n\nY axis is Win/Draw/Loss. X axis is Correct/Incorrect\n")
  #betMatrix
  
endfunction
