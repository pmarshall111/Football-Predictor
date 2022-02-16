%File to create predictions and store the result in the file given as an output argument.
%Call using: 'octave FootballTrainingProbabilitiesScorePredict.m thetaPath.csv gamesToPredictPath.csv predictions.csv'
addpath(fileparts(mfilename('fullpath'))); # adding path to functions in current directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../")); # adding path to functions in parent directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../betDecision")); # adding path to functions in betDecision directory

# First retrieve args from command
args=argv();
thetaPath=args{1};
gamesToPredictPath=args{2};
outputPath=args{3};

# Read in data
testSet = csvread(gamesToPredictPath);
testX = testSet(:, 17:end);
bookieOdds = testSet(:, 1:3);
bookieProbabilities = 1./bookieOdds;
gameIds = testSet(:,15:15);

# Predict
all_theta = csvread(thetaPath);
[fullPredictions max] = predictOneVsAll(all_theta, testX);
probabilities = convertLogitsToProbability(fullPredictions);
y = zeros(size(testSet,1),1);
[_, __, ___, ____, _____, ______, resultsToBetOn] = kellyCriterion(bookieProbabilities, probabilities, y, 0.1, 0.2);
[_, __, ___, ____, _____, ______, resultsToLayBetOn] = kellyCriterionLay(bookieProbabilities, probabilities, y, 0.1, 0.05);

# Subtract 1 from the outcome. Octave uses 1 as min index whereas Java uses 0.
notMinus1 = resultsToBetOn(:,1:1)!=-1;
resultsToBetOn = [resultsToBetOn(:,1:1)-notMinus1,resultsToBetOn(:,2:end)];

notMinus1Lay = resultsToLayBetOn(:,1:1)!=-1;
resultsToLayBetOn = [resultsToLayBetOn(:,1:1)-notMinus1Lay,resultsToLayBetOn(:,2:end)];

# Add the game id to the predictions, as well as any recommendations for results
outputMatrix = [gameIds, probabilities, resultsToBetOn, resultsToLayBetOn]
dlmwrite(outputPath, outputMatrix, "delimiter", ",", "newline", "\n");
