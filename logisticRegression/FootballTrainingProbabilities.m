addpath(fileparts(mfilename('fullpath'))); # adding path to functions in current directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../")); # adding path to functions in parent directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../betDecision")); # adding path to functions in betDecision directory

lambda = 40; %hard coded lambda. We don't want function as that hides all the variables we create within it's scope.
fprintf("Training for lambda value of %f. Reading in data.\n", lambda)

trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/31Jan22/train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/31Jan22/nolineups_train.csv"));
X = trainingSet(:, 9:end);
probabilityOfResults = trainingSet(:, 6:6);
y = trainingSet(:, 8:8) .+ 1; # Java stores Home win as 0. Add 1 to all results.
trainingSetSize = size(trainingSet,1)

testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/31Jan22/test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/31Jan22/nolineups_test.csv"));
testX = testSet(:, 9:end);
testBookieOdds = testSet(:, 1:3);
testBookieProbs = 1./testBookieOdds;
testY = testSet(:, 8:8) .+ 1;

num_labels = 3;

fprintf('\nTraining One-vs-All Logistic Regression...')
[all_theta] = oneVsAllWithProbabilities(X, y, num_labels, lambda, probabilityOfResults);


%% ================ Part 3: Predict for training set ================

rowsWithProbabilityOf1 = trainingSet(trainingSet(:,6:6)==1,:);
rowsWithProb1X = rowsWithProbabilityOf1(:,9:end);
rowsWithProb1Y = rowsWithProbabilityOf1(:, 8:8) .+ 1;
[fullPredictions max] = predictOneVsAll(all_theta, rowsWithProb1X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(max == rowsWithProb1Y)) * 100);

%% ================ Part 4: Predict for test set ================

[testFullPredictions testMax] = predictOneVsAll(all_theta, testX);

fprintf('\nTest Set Accuracy: %f\n', mean(double(testMax == testY)) * 100);

%% ================ Part 5: Work out money made ================

highestBy = 0; %has to be this much more likely than second highest outcome.
betterThanBookies = 0.2; %has to be this amount better than betters probabilities

ourProbs = convertLogitsToProbability(testFullPredictions);

fprintf("\n\Confusion Matrix showing distribution of correctly picked bets\n")
Confusion_Matrix(testBookieProbs, ourProbs, testY);

fprintf("\n\nKelly Criterion results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = kellyCriterion(testBookieProbs, ourProbs, testY, highestBy, betterThanBookies);
totalReturn, totalSpent, profit, percentageProfit, numbBets
fprintf("\n\nHighest Prob only && Better Than Betters by results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = BTB_VariableStake(testBookieProbs, ourProbs, testY, highestBy, betterThanBookies);
totalReturn, totalSpent, profit, percentageProfit, numbBets
