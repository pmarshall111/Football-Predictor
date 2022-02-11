function [theta, J_history] = gradientDescentMoneyBookieOdds(X, y, theta, alpha, num_iters, bookieOdds, finalRes)


# y will be the result.
# we're looking to bet on everything
# each update we do 1 train for each model
# we then calculate the profit if we bet on everything vs the profit. I guess this can help us to identify big wins
# y should be the max money we can make for each row. Therefore we calculate y each iteration. Multiplying £5 by the odds we provide in our predictions and then calculating the error to that.
# betting 100% of stake on the most likely outcome.

  stake = 1;

  % Initialize some useful values
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);

  for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
  hTheta = X * theta;
  error = hTheta - y;

  correctRes = convertVectorToMatrixWhereValIsIndex(finalRes);  
  correctRes = sum(correctRes .* bookieOdds,2);
  correctRes = featureNormalize(correctRes);
  
  error = error .* (sigmoid(correctRes)+1);

  xError = X' * error;

  delta = xError / m;
  
  theta = theta - alpha * delta;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

  end


end