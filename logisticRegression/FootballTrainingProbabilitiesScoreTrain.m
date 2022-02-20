%Train a model and store the result in the file given as an input argument.
%Call using: 'octave FootballTrainingProbabilitiesScoreProd.m theta_output_filename.csv'

addpath(fileparts(mfilename('fullpath'))); # adding path to functions in current directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../")); # adding path to functions in parent directory

lambda = 32;
fprintf("Training for lambda value of %f. Reading in data.", lambda)

trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/20FebScores/nolineups_train_final_model.csv"));
X = trainingSet(:, 17:end);
probabilityOfResults = trainingSet(:, 14:14);
y = trainingSet(:, 16:16) .+ 1;
scores = trainingSet(:, 4:5);

fprintf('\nTraining One-vs-All Logistic Regression...')
[all_theta] = oneVsAll(X, y, 3, lambda);

args=argv();
outputFile=args{1};

dlmwrite(outputFile, all_theta, "delimiter", ",", "newline", "\n");
