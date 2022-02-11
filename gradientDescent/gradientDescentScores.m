function [theta, J_history] = gradientDescentScores(X, y, theta, alpha, num_iters, goalsAway)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
goalsAway = sigmoid(abs(goalsAway))+1;


% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

hTheta = X * theta;
error = hTheta - y .* goalsAway;

%size(X)
%size(error)

xError = X' * error;

delta = xError / m;

theta = theta - alpha * delta;





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMultiScores(X, y, theta, goalsAway);

end

end
