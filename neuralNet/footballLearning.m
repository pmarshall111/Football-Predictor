%Adding paths
addpath(fileparts(mfilename('fullpath'))); # adding path to functions in current directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../")); # adding path to functions in parent directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../betDecision")); # adding path to functions in betDecision directory

#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_short_train.csv"));
trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_short_no_odds_train.csv"));
X = trainingSet(:, 17:end);
#X = [ones(size(X,1),1), X];
probabilityOfResults = trainingSet(:, 14:14);
#X = [trainingSet(:, 11:11) X]; #Using 538 features.
y = trainingSet(:, 16:16) .+ 1; # Java stores Home win as 0. Add 1 to all results.
simulatedProbs = trainingSet(:, 8:10);
simulatedProbs = normaliseRowsToSumTo1(simulatedProbs);
scores = trainingSet(:, 4:5);
trainingSetSize = size(trainingSet,1)


%
% Initialising theta values
input_layer_size = size(X, 2);
hidden_layer_size = input_layer_size;
num_labels = 3;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%Start training
iterations = 100;
options = optimset('MaxIter', iterations);
lambda = 40;

fprintf("\n Iterations: %f. Lambda: %f\n", iterations, lambda);

% Create "short hand" for the cost function to be minimized   
costFunctionVect = @(p) nnCostFunctionVectorised(p, ...
  input_layer_size, ...
  hidden_layer_size, ...
  num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunctionVect, initial_nn_params, options);


% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
rowsWithProbabilityOf1 = trainingSet(trainingSet(:,14:14)==1,:);
rowsWithProb1X = rowsWithProbabilityOf1(:,17:end);
rowsWithProb1Y = rowsWithProbabilityOf1(:, 16:16) .+ 1;
[maxIdx maxVal] = predict(Theta1, Theta2, rowsWithProb1X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(maxIdx == rowsWithProb1Y)) * 100);


% Test model on unseen games
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_short_test.csv"));
testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_short_no_odds_test.csv"));
testX = testSet(:, 17:end);
#testX = [ones(size(testX,1),1), testX];
#testX = [testSet(:, 11:11) testX];
testBookieOdds = testSet(:, 1:3);
testBookieProbs = 1./testBookieOdds;
testY = testSet(:, 16:16) .+ 1;

bookieProbs = 1 ./ testSet(:, 1:3);
[maxProb, maxIdx] = max(bookieProbs, [], 2);
fprintf('\nBookie Accuracy: %f\n', mean(double(maxIdx == testY)) * 100);

[pred2, probabilities] = predict(Theta1, Theta2, testX);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred2 == testY)) * 100);

%Normalise probabilities
regProbs = regProbabilities(probabilities);

testError = meanSquaredError(regProbs, testSet(:, 8:10));
fprintf('\nTest set Mean Squared Error to simulated probabilities: %f\n', testError);

highestBy = 0.1;
betterThanBookies = 0.2;

fprintf("\n\Confusion Matrix showing distribution of correctly picked bets\n")
Confusion_Matrix(testBookieProbs, regProbs, testY);

fprintf("\n\nKelly Criterion results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = kellyCriterion(testBookieProbs, regProbs, testY, highestBy, betterThanBookies);
totalReturn, totalSpent, profit, percentageProfit, numbBets
fprintf("\n\nHighest Prob only && Better Than Betters by results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = BTB_VariableStake(testBookieProbs, regProbs, testY, highestBy, betterThanBookies);
totalReturn, totalSpent, profit, percentageProfit, numbBets

