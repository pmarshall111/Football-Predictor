function [J, grad] = lrCostFunctionWithProbabilitiesScore(theta, X, y, goalsAwayFromCorrectPrediction, probabilityOfResults, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


thetaX = X * theta;
hThetaX = sigmoid(thetaX);

newTheta = theta(:);
newTheta(1) = 0;

goalsAwayFromCorrectPrediction = sigmoid(goalsAwayFromCorrectPrediction)+1;

firstPart = y .* log(hThetaX) .* probabilityOfResults .* goalsAwayFromCorrectPrediction;
secondPart = (1-y) .* log(1 - hThetaX) .* probabilityOfResults .* goalsAwayFromCorrectPrediction;


regularization = sum(newTheta .^ 2) * lambda / (2*m);

J = -1 / m * sum(firstPart + secondPart) + regularization;

% =============================================================

#grad = X' * ((hThetaX - y));
grad = (X)' * ((hThetaX - y).* probabilityOfResults .* goalsAwayFromCorrectPrediction);
grad = 2*grad/m + 2*lambda/m * newTheta;

end
