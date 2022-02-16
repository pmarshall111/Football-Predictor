addpath(fileparts(mfilename('fullpath'))); # adding path to functions in current directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../")); # adding path to functions in parent directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../betDecision")); # adding path to functions in betDecision directory
clear;
lambda = 40; %hard coded lambda. We don't want function as that hides all the variables we create within it's scope.
fprintf("Training for lambda value of %f. Reading in data.\n", lambda)

#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebSimMatchesByScore/train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebSimMatchesByScore/nolineups_train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebSimMatchesByScore/nolineups_short_train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebSimMatchesByScore/nolineups_short_no_odds_train.csv"));
trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/12FebScores/nolineups_train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/12FebScores/train.csv"));
X = trainingSet(:, 17:end);
#X = [trainingSet(:, 11:11) X];
probabilityOfResults = trainingSet(:, 14:14);
y = trainingSet(:, 16:16) .+ 1; # Java stores Home win as 0. Add 1 to all results.
scores = trainingSet(:, 4:5);
trainingSetSize = size(trainingSet,1)

#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebSimMatchesByScore/test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebSimMatchesByScore/nolineups_test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebSimMatchesByScore/nolineups_short_test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebSimMatchesByScore/nolineups_short_no_odds_test.csv"));
testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/12FebScores/nolineups_test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/12FebScores/test.csv"));
testX = testSet(:, 17:end);
#testX = [testSet(:, 11:11) testX];
testBookieOdds = testSet(:, 1:3);
testBookieProbs = 1./testBookieOdds;
testY = testSet(:, 16:16) .+ 1;

num_labels = 3;

fprintf('\nTraining One-vs-All Logistic Regression...')
[all_theta] = oneVsAllWithProbabilitiesScore(X, y, num_labels, lambda, probabilityOfResults, scores);


%% ================ Part 3: Predict for training set ================

rowsWithProbabilityOf1 = trainingSet(trainingSet(:,14:14)==1,:);
rowsWithProb1X = rowsWithProbabilityOf1(:,17:end);
#rowsWithProb1X = [rowsWithProbabilityOf1(:, 11:11) rowsWithProb1X];
rowsWithProb1Y = rowsWithProbabilityOf1(:, 16:16) .+ 1;
[fullPredictions maxIdx] = predictOneVsAll(all_theta, rowsWithProb1X);
ourPreditionWinDrawLoss = getWinsDrawsLosses(maxIdx);
actualWinDrawLoss = getWinsDrawsLosses(rowsWithProb1Y);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(maxIdx == rowsWithProb1Y)) * 100);

%% ================ Part 4: Predict for test set ================

bookieProbs = 1 ./ testSet(:, 1:3);
[maxProb, maxIdx] = max(bookieProbs, [], 2);
fprintf('\nBookie Accuracy: %f\n', mean(double(maxIdx == testY)) * 100);

[testFullPredictions testMax] = predictOneVsAll(all_theta, testX);
ourProbs = convertLogitsToProbability(testFullPredictions);

fprintf('\nTest Set Accuracy: %f\n', mean(double(testMax == testY)) * 100);

testError = meanSquaredError(ourProbs, testSet(:, 8:10));
fprintf('\nTest set Mean Squared Error to simulated probabilities: %f\n', testError);

%% ================ Part 5: Work out money made ================

highestBy = 0.1; %has to be this much more likely than second highest outcome.
betterThanBookiesBy = 0.2; %has to be this amount better than betters probabilities

fprintf("\n\Confusion Matrix showing distribution of correctly picked bets\n")
Confusion_Matrix(testBookieProbs, ourProbs, testY);

fprintf("\n\nKelly Criterion results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = kellyCriterion(testBookieProbs, ourProbs, testY, highestBy, betterThanBookiesBy);
totalReturn, totalSpent, profit, percentageProfit, numbBets
analyseBets(resultsToBetOn);

fprintf("\n\nHighest Prob only && Better Than Betters by results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = BTB_VariableStake(testBookieProbs, ourProbs, testY, highestBy, betterThanBookiesBy);
totalReturn, totalSpent, profit, percentageProfit, numbBets
analyseBets(resultsToBetOn);

fprintf("\n\nKelly Criterion lay results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = kellyCriterionLay(testBookieProbs, ourProbs, testY, highestBy, 0.05);
totalReturn, totalSpent, profit, percentageProfit, numbBets
analyseBets(resultsToBetOn);