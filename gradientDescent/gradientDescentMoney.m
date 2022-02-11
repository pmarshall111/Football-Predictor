function [all_theta, J_history, lowest_theta] = gradientDescentMoney(X, y, homeTheta, drawTheta, awayTheta, alpha, num_iters)


lowest_cost_so_far = 100000000000000000000;
lowest_theta = [];

# y will be the result.
# we're looking to bet on everything
# each update we do 1 train for each model
# we then calculate the profit if we bet on everything vs the profit. I guess this can help us to identify big wins
# y should be the max money we can make for each row. Therefore we calculate y each iteration. Multiplying £5 by the odds we provide in our predictions and then calculating the error to that.
# betting 100% of stake on the most likely outcome.

  stake = 1;

  % Initialize some useful values
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 3);

  for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    # first create predictions. hTheta = the prediction.
    homeHTheta = X * homeTheta;
    drawHTheta = X * drawTheta;
    awayHTheta = X * awayTheta;

    # normalise predictions
    normalisedPreds = normaliseRowsToSumTo1([homeHTheta, drawHTheta, awayHTheta]);
    [maxVal, maxIdx] = max(normalisedPreds, [], 2);
    
    # calc the max money we can make with these predictions
    normalisedOdds = 1 ./ normalisedPreds;
    
    correctRes = convertVectorToMatrixWhereValIsIndex(y);  
    optimalMonies = correctRes .* stake .* normalisedOdds; #optimal money made

    # calc the error
    
    ourRes = convertVectorToMatrixWhereValIsIndex(maxIdx);
    while size(ourRes,2) < 3
      ourRes = [ourRes, zeros(size(ourRes, 1), 1)];
    endwhile
    moneyIn = correctRes .* ourRes .* stake .* normalisedOdds;
    moneyOut = (ourRes != correctRes) .* (ourRes == 1) .* stake;
    error = moneyIn - optimalMonies - moneyOut;
    
    homeError = error(:,1:1);
    drawError = error(:,2:2);
    awayError = error(:,3:3);
      
    # do the updates
    homeXError = X' * homeError;
    drawXError = X' * drawError;
    awayXError = X' * awayError;

    homeTheta = homeTheta - alpha * homeXError/m;
    drawTheta = drawTheta - alpha * drawXError/m;
    awayTheta = awayTheta - alpha * awayXError/m;



    % ============================================================

    % Save the cost J in every iteration    
    
    home_J = sum(homeError .^2) /(2*m);
    draw_J = sum(drawError .^2) /(2*m);
    away_J = sum(awayError .^2) /(2*m);
    J_history(iter,:) = [home_J, draw_J, away_J];
    
    total_cost = home_J + draw_J + away_J;
    if total_cost < lowest_cost_so_far
      lowest_theta = [homeTheta drawTheta awayTheta];
    endif

  end

  all_theta = [homeTheta drawTheta awayTheta];

end
