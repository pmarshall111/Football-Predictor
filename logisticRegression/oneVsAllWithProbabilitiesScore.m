function [all_theta] = oneVsAllWithProbabilitiesScore(X, y, num_labels, lambda, probabilityOfResults, scores)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1); %5000
n = size(X, 2); %400

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);


goalsAwayFromHomeWin = zeros(m,1);
goalsAwayFromDraw = zeros(m,1);
goalsAwayFromAwayWin = zeros(m,1);

for i = 1:size(scores,1)
  homeScore = scores(i,1);
  awayScore = scores(i,2);
  
  if homeScore > awayScore
    goalsAwayFromDraw(i) = homeScore - awayScore;
    goalsAwayFromAwayWin(i) = homeScore - awayScore + 1; # plus 1 is because the goal diff is the amount to get to a draw. to get to a win, we need to +1
    goalsAwayFromHomeWin(i) = 1 - (homeScore-awayScore);
  elseif homeScore == awayScore
    goalsAwayFromHomeWin(i) = 1;
    goalsAwayFromAwayWin(i) = 1;
    goalsAwayFromDraw(i) = 0;
  else
    goalsAwayFromHomeWin(i) = awayScore - homeScore + 1;
    goalsAwayFromDraw(i) = awayScore - homeScore;
    goalsAwayFromAwayWin(i) = 1 - (awayScore - homeScore);
  endif
endfor



# Logistic Regression for a home win
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 5000);
[theta] = fmincg (@(t)(lrCostFunctionWithProbabilitiesScore(t, X, (y == 1), abs(goalsAwayFromHomeWin), probabilityOfResults, lambda)), initial_theta, options);      
all_theta(1,:) = theta;

# Logistic Regression for a draw
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 5000);
[theta] = fmincg (@(t)(lrCostFunctionWithProbabilitiesScore(t, X, (y == 2), abs(goalsAwayFromDraw), probabilityOfResults, lambda)), initial_theta, options);      
all_theta(2,:) = theta;

# Logistic Regression for an away win
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 5000);
[theta] = fmincg (@(t)(lrCostFunctionWithProbabilitiesScore(t, X, (y == 3), abs(goalsAwayFromAwayWin), probabilityOfResults, lambda)), initial_theta, options);      
all_theta(3,:) = theta;




% =========================================================================


end
