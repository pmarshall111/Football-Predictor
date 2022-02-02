%Adding paths
addpath(fileparts(mfilename('fullpath'))); # adding path to functions in current directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../")); # adding path to functions in parent directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../betDecision")); # adding path to functions in betDecision directory

% load in training data.
trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/23Jan22/toTrainOn_extended.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/23Jan22/nolineups_toTrainOn_extended.csv"));
X = trainingSet(:, 6:end);
y = trainingSet(:, 5:5) .+ 1;



%
% Initialising theta values
hInput_layer_size = size(X, 2);
hHidden_layer_size = hInput_layer_size * 2;
num_labels = 3;

hInitial_Theta1 = randInitializeWeights(hInput_layer_size, hHidden_layer_size);
hInitial_Theta2 = randInitializeWeights(hHidden_layer_size, hHidden_layer_size);
hInitial_Theta3 = randInitializeWeights(hHidden_layer_size, num_labels);


% Unroll parameters
hInitial_nn_params = [hInitial_Theta1(:) ; hInitial_Theta2(:); hInitial_Theta3(:)];

%starting training
iterations = 50;
options = optimset('MaxIter', iterations);
lambda = 40;

fprintf("\n Iterations: %f. Lambda: %f\n", iterations, lambda);

% Create "short hand" for the cost function to be minimized
costFunction2Hidden = @(p) nnCostFunction2Hidden(p, ...
                                   hInput_layer_size, ...
                                   hHidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[hNn_params, hCost] = fmincg(costFunction2Hidden, hInitial_nn_params, options);


% Obtain Thetas back from nn_params
hTheta1Size = hHidden_layer_size * (hInput_layer_size + 1);
hTheta2Size = hHidden_layer_size * (hHidden_layer_size + 1);

hTheta1 = reshape(hNn_params(1:hTheta1Size), ...
                 hHidden_layer_size, (hInput_layer_size + 1));

hTheta2 = reshape(hNn_params((1 + hTheta1Size) : hTheta1Size + hTheta2Size), ...
                 hHidden_layer_size, (hHidden_layer_size + 1));
                 
hTheta3 = reshape(hNn_params(1 + hTheta1Size + hTheta2Size : end),
                  num_labels, hHidden_layer_size+1);

                 
[hPred, actualVals] = predict2Hidden(hTheta1, hTheta2, hTheta3, X);
fprintf('\nTraining Set Accuracy 2 hidden layers: %f\n', mean(double(hPred == y)) * 100);


% Test model on unseen games
testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/23Jan22/lastSeason_extended.csv"));
#testSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/23Jan22/nolineups_lastSeason_extended.csv"));
testX = testSet(:, 6:end);
testBookieOdds = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/23Jan22/oddslastSeason_extended.csv"));
#testBookieOdds = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/23Jan22/oddsnolineups_lastSeason_extended.csv"));
testBookieProbs = 1./testBookieOdds;
testY = testSet(:, 5:5) .+ 1;

[pred2, probabilities] = predict2Hidden(hTheta1, hTheta2, hTheta3, testX);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred2 == testY)) * 100);

%need to normalise probabilities before we pass into calcMoneyMade function.
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
