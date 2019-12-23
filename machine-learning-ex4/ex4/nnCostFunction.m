function [J grad] = nnCostFunction(nn_params, ...
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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

% Initialise a
a_1 = X;
z_2 = zeros(hidden_layer_size, 1);
a_2 = zeros(hidden_layer_size, 1);
z_3 = zeros(num_labels, 1);
a_3 = zeros(num_labels, 1);

% Compute a
function computeActivationFunction
    a_1 = [ones(m, 1) X];
    z_2 = a_1 * Theta1';
    a_2 = [ones(m, 1) sigmoid(z_2)];
    z_3 = a_2 * Theta2';
    a_3 = sigmoid(z_3);
endfunction

% Converts y (size = [m, 1]), where each entry in y is in [1, k]
% to label (size = [m, k])
function y_v = convertLabelsToVectors(y)
    % Logical arrays
    y_v = [1:num_labels] == y; 
endfunction

function computeCost
    computeActivationFunction;
    y_v = convertLabelsToVectors(y);

    % cost 
    c_each = -y_v .* log(a_3) - (1 - y_v) .* log(1 - a_3); % size = [m, s_L]
    c_all = (1 / m) * sum(sum(c_each));

    % regularisation term - rmb to not regularise bias terms!


    Theta1_squared = Theta1 .^ 2;
    Theta2_squared = Theta2 .^ 2;
    Theta1_squared_wo_bias = Theta1_squared(:, 2:end);
    Theta2_squared_wo_bias = Theta2_squared(:, 2:end);
    r = (lambda / (2 * m)) * ...
        (sum(sum(Theta1_squared_wo_bias)) + sum(sum(Theta2_squared_wo_bias)));

    % Cost 
    J = c_all + r;
endfunction

function backprop 
    % Step 1: Compute a 
    computeActivationFunction;
    
    % Step 2: Compute error terms
    y_v = convertLabelsToVectors(y);
    d_3 = a_3 - y_v; % size = [5000, 10] 

    % Step 3: Delta 2
    d_2 = (d_3 * Theta2)(:, 2: end) .* sigmoidGradient(z_2); % size = [5000, 25]; remember to remove first error term

    % Step 4 and 5: Accumulate gradient and divide by sample size        
    Theta1_grad = (1 / m) .* (d_2' * a_1); % size = [25, 401]
    Theta2_grad = (1 / m) .* (d_3' * a_2); % size = [10, 25]

endfunction

function regularise
    r_1 = (lambda / m) .* (Theta1);
    r_1(:, 1) = 0; % Don't regularise bias term

    r_2 = (lambda / m) .* (Theta2);
    r_2(:, 1) = 0; % Don't regularise bias term

    Theta1_grad = Theta1_grad + r_1;
    Theta2_grad = Theta2_grad + r_2;
endfunction

computeCost;
backprop;
regularise;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
