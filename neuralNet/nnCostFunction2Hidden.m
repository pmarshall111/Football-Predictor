function [J grad] = nnCostFunction2Hidden(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1Size = hidden_layer_size * (input_layer_size + 1);
Theta2Size = hidden_layer_size * (hidden_layer_size + 1);

Theta1 = reshape(nn_params(1:Theta1Size), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params(1 + Theta1Size : Theta1Size + Theta2Size), ...
                 hidden_layer_size, (hidden_layer_size + 1));
                 
Theta3 = reshape(nn_params(1 + Theta1Size + Theta2Size : end),
                  num_labels, hidden_layer_size+1);

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y_m = zeros(size(y,1),num_labels);
for i = 1:size(y_m,1)
    y_m(i,y(i)) = 1;
end

Xn = [ones(size(X,1),1) X];
z2 = Xn * Theta1';
a2 = sigmoid(z2);
z3 = [ones(size(a2,1),1) a2] * Theta2';
a3 = sigmoid(z3);
z4 = [ones(size(a3,1),1) a3] * Theta3';
a4 = sigmoid(z4);

J = sum(sum(-y_m .* log(a4) .- (1-y_m) .* log(1-a4))) / size(y_m,1);
J = J + (sum(sum((Theta1 .* Theta1)(:,2:end))) + sum(sum((Theta2 .* Theta2)(:,2:end))) + sum(sum((Theta3 .* Theta3)(:,2:end))))...
 * lambda/ (2 * size(y_m,1)); %regularization

for t=1:m
    a1 = Xn(t,:);
    z2 = a1 * Theta1';
    a2 = [1 sigmoid(z2)];
    z3 = a2 * Theta2';
    a3 = [1 sigmoid(z3)];
    z4 = a3 * Theta3';
    a4 = sigmoid(z4);


    d4 = a4' - y_m(t,:)';
    d3 = (Theta3' * d4)(2:end) .* sigmoidGradient(z3)';
    d2 = (Theta2' * d3)(2:end) .* sigmoidGradient(z2)';
    
    Theta3_grad = Theta3_grad + d4 * a3;
    Theta2_grad = Theta2_grad + d3 * a2;
    Theta1_grad = Theta1_grad + d2 * a1;
end

Theta1_grad = 1/m * Theta1_grad;
Theta2_grad = 1/m * Theta2_grad;
Theta3_grad = 1/m * Theta3_grad;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);
Theta3_grad(:,2:end) = Theta3_grad(:,2:end) + lambda/m * Theta3(:,2:end);

grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];
%delta3 = (a3 -y_m)' *[ones(size(a2,1),1) a2 ];
%size(delta3)
%delta2 = sum((Theta2 .* delta3)(:,2:end) *  (sigmoidGradient(z2)'));
%size(delta2)