function Confusion_Matrix(betProbs, ourProbs, y)

  win_matrix = [0,0; 0,0];
  draw_matrix = [0,0; 0,0];
  loss_matrix = [0,0; 0,0];
  
  for row = 1:size(ourProbs,1)
    
    # find highest index
    highestProb = 0;
    highestIndex = -1;
    for col = 1:size(ourProbs,2)
      currProb = ourProbs(row,col);
      if currProb > highestProb
        highestProb = currProb;
        highestIndex = col;
      endif
    endfor
    
    #loop through columns and set true positive/false positive etc
    outcome = y(row);
    
    
    # win
    if highestIndex == 1
      if outcome == 1
        win_matrix(1,1) = win_matrix(1,1) + 1;
        draw_matrix(2,2) = draw_matrix(2,2) + 1;
        loss_matrix(2,2) = loss_matrix(2,2) + 1;
      elseif outcome == 2
        win_matrix(1,2) = win_matrix(1,2) + 1;
        draw_matrix(2,1) = draw_matrix(2,1) + 1;
        loss_matrix(2,2) = loss_matrix(2,2) + 1;
      elseif outcome == 3
        win_matrix(1,2) = win_matrix(1,2) + 1;
        draw_matrix(2,2) = draw_matrix(2,2) + 1;
        loss_matrix(2,1) = loss_matrix(2,1) + 1;
      endif
      
    elseif highestIndex == 2
      if outcome == 1
        win_matrix(2,1) = win_matrix(2,1) + 1;
        draw_matrix(1,2) = draw_matrix(1,2) + 1;
        loss_matrix(2,2) = loss_matrix(2,2) + 1;
      elseif outcome == 2
        win_matrix(2,2) = win_matrix(2,2) + 1;
        draw_matrix(1,1) = draw_matrix(1,1) + 1;
        loss_matrix(2,2) = loss_matrix(2,2) + 1;
      elseif outcome == 3
        win_matrix(2,2) = win_matrix(2,2) + 1;
        draw_matrix(1,2) = draw_matrix(1,2) + 1;
        loss_matrix(2,1) = loss_matrix(2,1) + 1;
      endif
      
    elseif highestIndex == 3
      if outcome == 1
        win_matrix(2,1) = win_matrix(2,1) + 1;
        draw_matrix(2,2) = draw_matrix(2,2) + 1;
        loss_matrix(1,2) = loss_matrix(1,2) + 1;
      elseif outcome == 2
        win_matrix(2,2) = win_matrix(2,2) + 1;
        draw_matrix(2,1) = draw_matrix(2,1) + 1;
        loss_matrix(1,2) = loss_matrix(1,2) + 1;
      elseif outcome == 3
        win_matrix(2,2) = win_matrix(2,2) + 1;
        draw_matrix(2,2) = draw_matrix(2,2) + 1;
        loss_matrix(1,1) = loss_matrix(1,1) + 1;
      endif
    endif
    
  endfor
  
  win_matrix
  draw_matrix
  loss_matrix

endfunction