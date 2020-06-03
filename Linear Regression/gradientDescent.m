function [theta, J_history] = gradientDescent(X, y, theta, alpha, iterations)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, iterations) updates theta by 
%   taking iterations gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(iterations, 1);

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================

    % Save the cost J in every iteration    
for iter = 1:iterations
J_history=computeCost(X,y,theta);
sum=0;
for i=1:m
a=X(i,:)
a=a';
prediction=theta'*a;
z=prediction-y(i);
z=z*a;
sum=sum+z
end
delta=sum/m;
delta=alpha*delta;
theta=theta-delta;
end

