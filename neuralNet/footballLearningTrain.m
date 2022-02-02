%Train a model and store the result in the file given as an input argument.
%Call using: 'octave FootballTrainingProbabilitiesScoreProd.m theta_output_filename.csv'

addpath(fileparts(mfilename('fullpath'))); # adding path to functions in current directory
addpath(strcat(fileparts(mfilename('fullpath')),"/../")); # adding path to functions in parent directory

lambda = 160;
fprintf("Training for lambda value of %f. Reading in data.", lambda)

trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/28Jan22/train_final_model_score.csv"));
#trainingSet = csvread(strcat(fileparts(mfilename('fullpath')), "/../data/28Jan22/train_final_model_score.csv"));
X = trainingSet(:, 9:end);
probabilityOfResults = trainingSet(:, 6:6);
y = trainingSet(:, 8:8) .+ 1;
scores = trainingSet(:, 4:5);

fprintf('\nTraining Neural network...')
%
% Initialising theta values
input_layer_size = size(X, 2);
hidden_layer_size = input_layer_size;
num_labels = 3;

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%Start training
iterations = 2;
options = optimset('MaxIter', iterations);
lambda = 40;

fprintf("\n Iterations: %f. Lambda: %f\n", iterations, lambda);

% Create "short hand" for the cost function to be minimized   
costFunctionVect = @(p) nnCostFunctionVectorisedProbs(p, ...
  input_layer_size, ...
  hidden_layer_size, ...
  num_labels, X, y, lambda, probabilityOfResults, scores);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunctionVect, initial_nn_params, options);


% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

args=argv();
outputFile=args{1};

dlmwrite(outputFile, all_theta, "delimiter", ",", "newline", "\n");
