addpath(fileparts(mfilename('fullpath'))); # adding path to functions in current directory
addpath(strcat(fileparts(mfilename('fullpath')),"/betDecision")); # adding path to functions in betDecision directory

trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/data/02Feb22/train_final_model.csv"));
bookieProbs = 1 ./ trainingSet(:, 1:3);
simulatedProbs = trainingSet(:, 8:10);
simulatedProbs = normaliseRowsToSumTo1(simulatedProbs);
y = trainingSet(:, 16:16) .+ 1;

[maxProb, maxIdx] = max(simulatedProbs, [], 2);
fprintf('\nSimulated Probabilities Accuracy: %f\n', mean(double(maxIdx == y)) * 100);

fprintf("\n\Confusion Matrix showing distribution of correctly picked bets\n")
Confusion_Matrix(bookieProbs, simulatedProbs, y);

highestBy = 0;
betterThanBookies = 0.2;

fprintf("\n\nKelly Criterion results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = kellyCriterion(bookieProbs, simulatedProbs, y, highestBy, betterThanBookies);
totalReturn, totalSpent, profit, percentageProfit, numbBets
fprintf("\n\nHighest Prob only && Better Than Betters by results\n")
[totalReturn, totalSpent, profit, percentageProfit, numbBets, betMatrix, resultsToBetOn] = BTB_VariableStake(bookieProbs, simulatedProbs, y, highestBy, betterThanBookies);
totalReturn, totalSpent, profit, percentageProfit, numbBets
