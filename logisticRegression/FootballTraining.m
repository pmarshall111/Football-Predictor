addpath(fileparts(mfilename('fullpath'))); # adding path to functions in current directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../")); # adding path to functions in parent directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../betDecision")); # adding path to functions in betDecision directory

lambda = 4; %hard coded lambda. We don't want function as that hides all the variables we create within it's scope.
fprintf("Training for lambda value of %f. Reading in data.\n", lambda)

#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/12FebBase/nolineups_train.csv"));
trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/13FebBase/train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_short_train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_short_no_odds_train.csv"));
X = trainingSet(:, 17:end);
X = [trainingSet(:, 11:11) X]; #Using 538 features.
y = trainingSet(:, 16:16) .+ 1; # Java stores Home win as 0. Add 1 to all results.
trainingSetSize = size(trainingSet)

#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/test.csv"));
testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/13FebBase/test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/12FebBase/nolineups_test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_short_test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_short_no_odds_test.csv"));
testX = testSet(:, 17:end);
testX = [testSet(:, 11:11) testX];
testBookieOdds = testSet(:, 1:3);
testBookieProbs = 1./testBookieOdds;
testY = testSet(:, 16:16) .+ 1;
fiveThirtyEightProbs = testSet(:, 11:13);

num_labels = 3;

fprintf('\nTraining One-vs-All Logistic Regression...')
[all_theta] = oneVsAll(X, y, num_labels, lambda);


%% ================ Part 3: Predict for training set ================

[fullPredictions maxIdx] = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(maxIdx == y)) * 100);

%% ================ Part 4: Predict for test set ================

bookieProbs = 1 ./ testSet(:, 1:3);
[maxProb, maxIdx] = max(bookieProbs, [], 2);
fprintf('\nBookie Accuracy: %f\n', mean(double(maxIdx == testY)) * 100);

[maxProb, maxIdx] = max(fiveThirtyEightProbs, [], 2);
fprintf('\FiveThirtyEight Accuracy: %f\n', mean(double(maxIdx == testY)) * 100);

[testFullPredictions testMax] = predictOneVsAll(all_theta, testX);

fprintf('\nTest Set Accuracy: %f\n', mean(double(testMax == testY)) * 100);
ourProbs = convertLogitsToProbability(testFullPredictions);
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