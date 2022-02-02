addpath(fileparts(mfilename('fullpath'))); # adding path to functions in current directory

trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/data/01cFeb22/train_final_model.csv"));
bookieProbs = 1 ./ trainingSet(:, 1:3);
y = trainingSet(:, 13:13) .+ 1;

[maxProb, maxIdx] = max(bookieProbs, [], 2);
fprintf('\nBookie Accuracy: %f\n', mean(double(maxIdx == y)) * 100);