%Adding paths
addpath(fileparts(mfilename('fullpath'))); # adding path to functions in current directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../")); # adding path to functions in parent directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../betDecision")); # adding path to functions in betDecision directory


trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/31Jan22/train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/31Jan22/nolineups_train.csv"));
X = trainingSet(:, 9:end);
probabilityOfResults = trainingSet(:, 6:6);
y = trainingSet(:, 8:8) .+ 1; # Java stores Home win as 0. Add 1 to all results.
trainingSetSize = size(trainingSet,1)
scores = trainingSet(:, 1:2);

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
costFunctionVect = @(p) nnCostFunctionVectorisedProbs(p, ...
  input_layer_size, ...
  hidden_layer_size, ...
  num_labels, X, y, lambda,probabilityOfResults, scores);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunctionVect, initial_nn_params, options);


% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
                 
rowsWithProbabilityOf1 = trainingSet(trainingSet(:,6:6)==1,:);
rowsWithProb1X = rowsWithProbabilityOf1(:,9:end);
rowsWithProb1Y = rowsWithProbabilityOf1(:, 8:8) .+ 1;
[fullPredictions max] = predictOneVsAll(all_theta, rowsWithProb1X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(max == rowsWithProb1Y)) * 100);



testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/31Jan22/test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/31Jan22/nolineups_test.csv"));
testX = testSet(:, 9:end);
testBookieOdds = testSet(:, 1:3);
testBookieProbs = 1./testBookieOdds;
testY = testSet(:, 8:8) .+ 1;

[pred2, probabilities] = predict(Theta1, Theta2, testX);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred2 == testY)) * 100);

%Normalise probabilities
regProbs = regProbabilities(probabilities);

highestBy = 0;
betterThanBookies = 0.2;

fprintf("\n\Confusion Matrix showing distribution of correctly picked bets\n")
Confusion_Matrix(testBookieProbs, regProbs, testY);

fprintf("\n\nKelly Criterion results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = kellyCriterion(testBookieProbs, ourProbs, testY, highestBy, betterThanBookies);
totalReturn, totalSpent, profit, percentageProfit, numbBets
fprintf("\n\nHighest Prob only && Better Than Betters by results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = BTB_VariableStake(testBookieProbs, ourProbs, testY, highestBy, betterThanBookies);
totalReturn, totalSpent, profit, percentageProfit, numbBets
