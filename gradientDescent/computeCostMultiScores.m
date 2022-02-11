function J = computeCostMultiScores(X, y, theta, goalsAway)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
goalsAway = sigmoid(goalsAway)+1;
% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

hTheta = X * theta;

error = hTheta - y .* goalsAway;
squaredError = error .^2;

J = sum(squaredError) /(2*m);

% =========================================================================

end