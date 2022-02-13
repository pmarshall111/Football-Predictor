function analyseBets(resultsToBetOn)
  # only look at rows with a bet
  notMinus1 = resultsToBetOn(:,1:1)!=-1;
  resultsToBetOn = resultsToBetOn(notMinus1,:);
  
  
  betsSorted = sortrows(resultsToBetOn, [-4]);
  
  betAccuracy = sum(betsSorted(:,4:4)>0) / length(betsSorted)
  
  
  l = length(betsSorted);
  # Get avg of middle 50% of data
  idx25 = floor(l/4);
  idx75 = floor(3*l/4);
  avgOfMiddle50Prec = mean(betsSorted(idx25:idx75,4:4))
  
  # Get avg of middle 75% of data
  idx12p5 = floor(l/8);
  idx87p5 = floor(7*l/8);
  avgOfMiddle75Prec = mean(betsSorted(idx12p5:idx87p5,4:4))
  
  totalAvg = mean(betsSorted(:,4:4))
  
  biggestWin = prctile(betsSorted(:,4:4),100)
  biggestLoss = prctile(betsSorted(:,4:4),0)

  medianBet = median(betsSorted(:,4:4))
  
  # Want a smaller standard deviation to give less varied results
  standardDeviation = std(betsSorted(:,4:4))
  
  # Expect a positive skewness. 
  skewnessAmount = skewness(betsSorted(:,4:4))
  
endfunction
