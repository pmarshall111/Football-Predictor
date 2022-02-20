function error = plotBets(resultsToBetOn) 
  notMinus1 = resultsToBetOn(:,1:1)!=-1;
  resultsToBetOn = resultsToBetOn(notMinus1,:);
  
  stairs(cumsum(resultsToBetOn(:,4:4)));
  #title("Linear Stake Betting Strategy");
  xlabel("Bet number");
  ylabel("Cumulative profit /£");
  
endfunction
