function [J grad] = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda)
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X=[ones(m,1) X];
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


for i=1:m
s=zeros(num_labels,1);
a1=(X(i,:)');
z2=Theta1*a1;
a2=sigmoid(z2);
a2=[1;a2];
z3=Theta2*a2;
pred=sigmoid(z3);
for k=1:num_labels
yi=zeros(num_labels,1);
yi(y(i,1))=1;
pr=pred(k,1);
p=-yi*log(pr);
n=(1-yi)*log(1-pr);
s(k,1)=s(k,1)+p(k,1)-n(k,1);
end
J=J+sum(s);
end
J=J/m;
s1=0;
s2=0;
theta1=Theta1(:,2:input_layer_size+1);
theta2=Theta2(:,2:hidden_layer_size+1);

for j=1:hidden_layer_size
for k=1:input_layer_size
    s1=s1+(theta1(j,k)^2);
end
end

for j=1:num_labels
for k=1:hidden_layer_size
    s2=s2+(theta2(j,k)^2);
end
end
J=J+((lambda/(2*m))*(s1+s2));

delta1=Theta1_grad;
delta2=zeros(num_labels,hidden_layer_size+1);
for t=1:m
yi=zeros(1:num_labels);
yi(y(t))=1;
a1=(X(t,:)');
z2=Theta1*a1;
a2=sigmoid(z2);
a2=[1;a2];
z3=Theta2*a2;
pred=sigmoid(z3);
for k=1:num_labels
d3(k,1)=pred(k,1)-yi(k);
end
d2=(Theta2(:,2:hidden_layer_size+1)'*d3).*sigmoidGradient(z2);
delta2=delta2+(d3*a2');
delta1=delta1+(d2*a1');
end
Theta1_grad=delta1/m;
Theta2_grad=delta2/m;
[o,p]=size(Theta1_grad);
q1=zeros(o,p);
for i=1:o
    for j=2:p
        q1(i,j)=(lambda/m)*Theta1(i,j);
    end
end
Theta1_grad=Theta1_grad+q1;
[o,p]=size(Theta2_grad);
q2=zeros(o,p);
for i=1:o
    for j=2:p
        q2(i,j)=(lambda/m)*Theta2(i,j);
    end
end
Theta2_grad=Theta2_grad+q2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
