function winDrawLoss = getWinsDrawsLosses(y)
  totalGames = size(y,1);
  wins = sum(sum(y==1))/totalGames;
  draws = sum(sum(y==2))/totalGames;
  loss = sum(sum(y==3))/totalGames;
  
  winDrawLoss = [wins, draws, loss];
endfunction
