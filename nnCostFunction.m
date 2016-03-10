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

%  Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
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

%fprintf('3')
%pause;
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
del1=0;
del2=0;
X1=horzcat(ones(m,1),X);
t1=power(Theta1,2);
t2=power(Theta2,2);
reg=(lambda/(2*m))*(sum(t1(:))+sum(t2(:))-sum(t1(:,1))-sum(t2(:,1)));
%fprintf('4')
%pause;
for i=1:m
    z1=(Theta1)*X1(i,:)';
H1=power((1+exp(-z1)),-1);

%fprintf('5')
%pause;
H1=vertcat(ones(1,1),H1);

%fprintf('6')
%pause;
z2=(Theta2)*H1;
H2=power((1+exp(-z2)),-1);

%fprintf('7')
%pause;
g=y(i,1);

%fprintf('8')
%pause;
a=zeros(1,num_labels);
if(y(i,1)>0)
    
        if g ==1
           
           a=horzcat(1,zeros(1,num_labels-1));
           
       
        elseif y(i,1)>1
           
           a=[zeros(1,y(i,1)-1),1,zeros(1,num_labels-y(i,1))]  ;
           
        end
        
           
    
elseif(y(i,1)==num_labels)
    
    a=[zeros(1,num_labels-1),1] ;
end
    

%pause;
%fprintf('9')
%pause;
p=-a*log(H2);
o=-(1-a)*log(1-H2); 

J=J+p+o;

d3=H2-a';
lm=(Theta2)'*d3;
d2=((Theta2)'*d3).*vertcat(ones(1,1),sigmoidGradient(z1));
d2=d2(2:end);
del1=del1+d2*X1(i,:);
del2=del2+d3*H1';
















% -------------------------------------------------------------

% =========================================================================
end
%Theta1_grad=del1/m;
%Theta2_grad=del2/m;
Theta1_grad(1:end,1)=del1(1:end,1)/m;
Theta1_grad(1:end,2:end)=del1(1:end,2:end)/m+lambda*(Theta1(1:end,2:end)/m);

Theta2_grad(1:end,1)=del2(1:end,1)/m;
Theta2_grad(1:end,2:end)=del2(1:end,2:end)/m+lambda*(Theta2(1:end,2:end)/m);
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



J=J/m+reg;

end

