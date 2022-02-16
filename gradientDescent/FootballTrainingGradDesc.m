addpath(fileparts(mfilename('fullpath'))); # adding path to functions in current directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../")); # adding path to functions in parent directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../betDecision")); # adding path to functions in betDecision directory

#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/12FebBase/nolineups_train.csv"));
trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/13FebBase/train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_short_train.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_short_no_odds_train.csv"));
X = trainingSet(:, 17:end-1);
X = [ones(size(X,1),1), X];
#X = [trainingSet(:, 11:11) X]; #Using 538 features.
y = trainingSet(:, 16:16) .+ 1; # Java stores Home win as 0. Add 1 to all results.
simulatedProbs = trainingSet(:, 8:10);
simulatedProbs = normaliseRowsToSumTo1(simulatedProbs);
trainBookieProbs = 1 ./ trainingSet(:,1:3);

trainingSetSize = size(trainingSet,1)

#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/12FebBase/nolineups_test.csv"));
testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/13FebBase/test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_short_test.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/08FebBase/nolineups_short_no_odds_test.csv"));
testX = testSet(:, 17:end-1);
testX = [ones(size(testX,1),1), testX];
#testX = [testSet(:, 11:11) testX];
testBookieOdds = testSet(:, 1:3);
testBookieProbs = 1./testBookieOdds;
testY = testSet(:, 16:16) .+ 1;

testSetSize = size(testSet,1)

## Train a gradient descent to predict the win, draw and loss probability. This is learned based on the probability produced by simulating the final XG via a poisson distribution
homeProb = simulatedProbs(:, 1:1);
drawProb = simulatedProbs(:, 2:2);
awayProb = simulatedProbs(:, 3:3);



alpha = 0.05;
num_iters = 10000;
thetaInit = zeros(1, size(X,2))';

[homeTheta, home_J_history] = gradientDescent(X, homeProb, zeros(1, size(X,2))', alpha, num_iters);
[drawTheta, draw_J_history] = gradientDescent(X, drawProb, zeros(1, size(X,2))', alpha, num_iters);
[awayTheta, away_J_history] = gradientDescent(X, awayProb, zeros(1, size(X,2))', alpha, num_iters);

# Plot costs of grad descent

figure;
plot(1:numel(home_J_history), home_J_history, '-b', 'LineWidth', 2);
hold on;
plot(1:numel(home_J_history), draw_J_history, '-r', 'LineWidth', 2);
plot(1:numel(home_J_history), away_J_history, '-k', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');


%% ================ Part 3: Predict for training set ================

rowsWithProbabilityOf1 = trainingSet(trainingSet(:,14:14)==1,:);
rowsWithProb1X = rowsWithProbabilityOf1(:,17:end-1);
rowsWithProb1Y = rowsWithProbabilityOf1(:, 16:16) .+ 1;
rowsWithProb1X = [ones(size(rowsWithProb1X,1),1), rowsWithProb1X];
#rowsWithProb1X = [rowsWithProbabilityOf1(:, 11:11) rowsWithProb1X];

trainPredictHome = rowsWithProb1X * homeTheta;
trainPredictDraw = rowsWithProb1X * drawTheta;
trainPredictAway = rowsWithProb1X * awayTheta;
#trainRegularisedPredictions = [trainPredictHome, trainPredictDraw, trainPredictAway];
trainRegularisedPredictions = normaliseRowsToSumTo1([trainPredictHome, trainPredictDraw, trainPredictAway]);
[trainMaxVal, trainMaxIdx] = max(trainRegularisedPredictions, [], 2);
[trainBookieMaxVal, trainBookieMaxIdx] = max(trainBookieProbs, [], 2);
ourPreditionWinDrawLoss = getWinsDrawsLosses(trainMaxIdx)
actualWinDrawLoss = getWinsDrawsLosses(rowsWithProb1Y)
bookieWinDrawLoss = getWinsDrawsLosses(trainBookieMaxIdx)

fprintf('\nTraining Set Accuracy: %f\n', mean(trainMaxIdx == rowsWithProb1Y) * 100);

%% ================ Part 4: Predict for test set ================


bookieProbs = 1 ./ testSet(:, 1:3);
[maxProb, maxIdx] = max(bookieProbs, [], 2);
fprintf('\nBookie Accuracy: %f\n', mean(double(maxIdx == testY)) * 100);


testPredictHome = testX * homeTheta;
testPredictDraw = testX * drawTheta;
testPredictAway = testX * awayTheta;
#testRegularisedPredictions = [testPredictHome, testPredictDraw, testPredictAway];
testRegularisedPredictions = normaliseRowsToSumTo1([testPredictHome, testPredictDraw, testPredictAway]);
[testMaxVal, testMaxIdx] = max(testRegularisedPredictions, [], 2);
fprintf('\nTest Set Accuracy: %f\n', mean(testMaxIdx == testY) * 100);

testError = meanSquaredError(testRegularisedPredictions, testSet(:, 8:10));
fprintf('\nTest set Mean Squared Error to simulated probabilities: %f\n', testError);


%% ================ Part 5: Work out money made ================

highestBy = 0.1; %has to be this much more likely than second highest outcome.
betterThanBookiesBy = 0.2; %has to be this amount better than betters probabilities

fprintf("\n\Confusion Matrix showing distribution of correctly picked bets\n")
Confusion_Matrix(testBookieProbs, testRegularisedPredictions, testY);

fprintf("\n\nKelly Criterion results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = kellyCriterion(testBookieProbs, testRegularisedPredictions, testY, highestBy, betterThanBookiesBy);
totalReturn, totalSpent, profit, percentageProfit, numbBets
analyseBets(resultsToBetOn);

fprintf("\n\nHighest Prob only && Better Than Betters by results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = BTB_VariableStake(testBookieProbs, testRegularisedPredictions, testY, highestBy, betterThanBookiesBy);
totalReturn, totalSpent, profit, percentageProfit, numbBets
analyseBets(resultsToBetOn);

fprintf("\n\nKelly Criterion lay results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = kellyCriterionLay(testBookieProbs, testRegularisedPredictions, testY, highestBy, 0.05);
totalReturn, totalSpent, profit, percentageProfit, numbBets
analyseBets(resultsToBetOn);