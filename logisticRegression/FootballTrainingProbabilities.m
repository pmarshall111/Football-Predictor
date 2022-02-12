addpath(fileparts(mfilename('fullpath'))); # adding path to functions in current directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../")); # adding path to functions in parent directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../betDecision")); # adding path to functions in betDecision directory

lambda = 4; %hard coded lambda. We don't want function as that hides all the variables we create within it's scope.
fprintf("Training for lambda value of %f. Reading in data.\n", lambda)

trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/12FebResult/nolineups_train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebSimMatchesByResult//nolineups_train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebSimMatchesByResult//nolineups_short_train.csv"));
X = trainingSet(:, 17:end);
#X = [trainingSet(:, 11:11) X]; #Using 538 features.
y = trainingSet(:, 16:16) .+ 1; # Java stores Home win as 0. Add 1 to all results.
trainingSetSize = size(trainingSet,1)
probabilityOfResults = trainingSet(:, 6:6);

testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/12FebResult/nolineups_test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebSimMatchesByResult//nolineups_test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebSimMatchesByResult//nolineups_short_test.csv"));
testX = testSet(:, 17:end);
#testX = [testSet(:, 11:11) testX];
testBookieOdds = testSet(:, 1:3);
testBookieProbs = 1./testBookieOdds;
testY = testSet(:, 16:16) .+ 1;

num_labels = 3;

fprintf('\nTraining One-vs-All Logistic Regression...')
[all_theta] = oneVsAllWithProbabilities(X, y, num_labels, lambda, probabilityOfResults);


%% ================ Part 3: Predict for training set ================

rowsWithProbabilityOf1 = trainingSet(trainingSet(:,14:14)==1,:);
rowsWithProb1X = rowsWithProbabilityOf1(:,17:end);
rowsWithProb1Y = rowsWithProbabilityOf1(:, 16:16) .+ 1;
#rowsWithProb1X = [rowsWithProbabilityOf1(:, 11:11) rowsWithProb1X];

[fullPredictions maxIdx] = predictOneVsAll(all_theta, rowsWithProb1X);
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
betterThanBookies = 0.2; %has to be this amount better than betters probabilities

fprintf("\n\Confusion Matrix showing distribution of correctly picked bets\n")
Confusion_Matrix(testBookieProbs, ourProbs, testY);

fprintf("\n\nKelly Criterion results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = kellyCriterion(testBookieProbs, ourProbs, testY, highestBy, betterThanBookies);
totalReturn, totalSpent, profit, percentageProfit, numbBets
fprintf("\n\nHighest Prob only && Better Than Betters by results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = BTB_VariableStake(testBookieProbs, ourProbs, testY, highestBy, betterThanBookies);
totalReturn, totalSpent, profit, percentageProfit, numbBets
fprintf("\n\nBetting on underdogs\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = betOnUnderdogs(testBookieProbs, ourProbs, testY, highestBy, betterThanBookies);
totalReturn, totalSpent, profit, percentageProfit, numbBets

fprintf("\n\nBetting on anything higher\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = betOnAnythingHigher(testBookieProbs, ourProbs, testY, betterThanBookies);
totalReturn, totalSpent, profit, percentageProfit, numbBets

fprintf("\n\nBetting on anything lower\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = betOnAnythingLower(testBookieProbs, ourProbs, testY, betterThanBookies);
totalReturn, totalSpent, profit, percentageProfit, numbBets

fprintf("\n\nBetting on close games\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = betOnCloseGamesBTB(testBookieProbs, ourProbs, testY, betterThanBookiesBy);
totalReturn, totalSpent, profit, percentageProfit, numbBets

fprintf("\n\nBetting on far apart games\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = betOnMismatchesBTB(testBookieProbs, ourProbs, testY, betterThanBookiesBy);
totalReturn, totalSpent, profit, percentageProfit, numbBets

fprintf("\n\nBetting on games where our favourite is not the bookies favourite\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = betOnDifferentFavourites(testBookieProbs, ourProbs, testY);
totalReturn, totalSpent, profit, percentageProfit, numbBets