% A simple neural network used to classify flowers into one of three categories based on four features: 
% (1) sepal length (2) sepal width (3) petal length (4) petal width 
% Author: Ruksana Kabealo

% Initialize data by loading it into an array (data has already been
% seperated into the training data and testing data in the .csv sheets

trainingData = xlsread('irisDataTrainingSheet2.csv');
testingData = xlsread('IDTest2.csv');

% Activation booleans, allows us to turn off and turn on the training and
% testing phases 
training = true; 
testing = true; 

% The data sheets are organized with the features in the first four columns
% and the classification in the last column 
% sLength = trainingData(:, 1);
% sWidth = trainingData(:, 2);
% pLength = trainingData(:, 3);
% pWidth = trainingData(:, 4);
% results = data(:,5-7);

%%%% TRAINING PHASE %%%%

% Create two matrices: one that holds all the weights between the input
% layer neurons and the hidden layer neurons, and another that holds all the
% weights between the hidden layer heurons and the output layer neurons. 
% We randomize the values in these matrices at first because we want to
% initially pick random weights. 
%hiddenWeights = rand(3,4); 
%outputWeights = rand(3,3); 

% Constant weights for debugging 
%hiddenWeights = [0.4074, 0.4567, 0.1392,0.4824; 0.4529,0.3162,0.2734,0.0788;0.0635,0.0488,0.4788,0.4853;];
%outputWeights = [0.4786,0.0709,0.3961;0.2427,0.2109,0.4797;0.4001,0.4579,0.3279;]

% BEST WEIGHTS. These are the weights that will give an output of 63.6667% accuracy
% (19/30 classified correctly) 
hiddenWeights =  [1.8831, 1.4256, 0.3347, 0.8075; -3.8684, -5.1074, 5.6607, 6.0634; 1.8402, 1.3170, 0.8307, 0.1677;]; 
outputWeights =  [183.0628, -72.6263,  -183.1733; -2.4803,-3.2581,2.3140; 182.4970,-72.1603 , -183.3125;];

% These are our biases for the hidden layer neurons and the output layer
% neurons, to be used in calculations later 
biasHidden = 0.2; 
biasOutput = 0.2; 

% Training phase, boolean activated. Trains with random, pre-associated data%

if(training == true)
    
disp("Initial hidden weights: ");
disp(hiddenWeights); 

disp("Initial output weights: ");
disp(outputWeights);

% Outer for loop is for repeated association. Inner for loop is for each
% sweep through the training data 
for i = 1:5
    
for t = 1:120
   
% Grab a single set of data: one flower described in terms of petal and
% sepal length and width (current inputs) and its corresponding flower type
% (current outputs). We want our program to output as close to a 1.0 as
% possible for the flower it catagorizes our data as, and as close to a 0.0
% as possible for flowers we know don't match our data 
currentInputs = trainingData(t,1:4)'; 
currentOutputs = trainingData(t,5:7)'; 

% Here we do the calculations to get the net inputs to all the
% neurons in the hidden layer 
netHidden = hiddenWeights*currentInputs - biasHidden;

% Using our sigmoid function, we smoosh down our values so they're strictly
% in the range [0.0,1.0]
outHidden = arrayfun(@sigmoid,netHidden); 

%  We store this array as a set of values for later
outH1 = outHidden(1,1); 
outH2 = outHidden(2,1); 
outH3 = outHidden(3,1); 

% Here we do the calculations to get the inputs to all the
% neurons in the output layer 
netOutput = outputWeights * outHidden - biasOutput; 
% Same smooshing with the sigmoid function 
outOutput = arrayfun(@sigmoid,netOutput); 

% Here's the actual probability that our iteration is a member of each
% category at this point
aSetosa = outOutput(1,1);  
aVersicolor = outOutput(2,1); 
aVirginica = outOutput(3,1); 

% Here's the desired probability that we want for this iteration 
dSetosa = currentOutputs(1,1); 
dVersicolor = currentOutputs(2,1);
dVirginica = currentOutputs(3,1); 
  
    % Adjusts all of the 2nd layer weights (between hidden and output) 
     
    % Adjust all the weights stemming FROM Hidden Layer Neuron #1
    outputWeights(1,1) = adjust2ndLayer(outputWeights(1,1), dSetosa, aSetosa, outH1);
    outputWeights(1,2) = adjust2ndLayer(outputWeights(1,2), dVersicolor, aVersicolor, outH1); 
    outputWeights(1,3) = adjust2ndLayer(outputWeights(1,3), dVirginica, aVirginica, outH1); 
    
    % Adjust all the weights stemming FROM Hidden Layer Neuron #2
    outputWeights(2,1) = adjust2ndLayer(outputWeights(2,1), dSetosa, aSetosa, outH2); 
    outputWeights(2,2) = adjust2ndLayer(outputWeights(2,2), dVersicolor, aVersicolor, outH2); 
    outputWeights(2,3) = adjust2ndLayer(outputWeights(2,3), dVirginica, aVirginica, outH2); 
    
    % Adjust all the weights stemming FROM Hidden Layer Neuron #3
    outputWeights(3,1) = adjust2ndLayer(outputWeights(3,1), dSetosa, aSetosa, outH3); 
    outputWeights(3,2) = adjust2ndLayer(outputWeights(3,2), dVersicolor, aVersicolor, outH3); 
    outputWeights(3,3) = adjust2ndLayer(outputWeights(3,3), dVirginica, aVirginica, outH3); 
    
    % Adjusts all of the 1st layer weights (between input and hidden) 
    
    % Adjust the weights that feed INTO Hidden Layer Neuron #1
    hiddenWeights(1,1) = adjust1stLayer(aSetosa, dSetosa, outputWeights(1,1), aVersicolor, dVersicolor, outputWeights(1,2), aVirginica, dVirginica, outputWeights(1,3),outH1, hiddenWeights(1,1), currentInputs(1,1)); 
    hiddenWeights(1,2) = adjust1stLayer(aSetosa, dSetosa, outputWeights(1,1), aVersicolor, dVersicolor, outputWeights(1,2), aVirginica, dVirginica, outputWeights(1,3),outH1, hiddenWeights(1,2),currentInputs(2,1)); 
    hiddenWeights(1,3) = adjust1stLayer(aSetosa, dSetosa, outputWeights(1,1), aVersicolor, dVersicolor, outputWeights(1,2), aVirginica, dVirginica, outputWeights(1,3),outH1, hiddenWeights(1,3),currentInputs(3,1)); 
    hiddenWeights(1,4) = adjust1stLayer(aSetosa, dSetosa, outputWeights(1,1), aVersicolor, dVersicolor, outputWeights(1,2), aVirginica, dVirginica, outputWeights(1,3),outH1, hiddenWeights(1,4), currentInputs(4,1)); 
   
    % Adjust the weights that feed INTO Hidden Layer Neuron #2
    hiddenWeights(2,1) = adjust1stLayer(aSetosa, dSetosa, outputWeights(2,1), aVersicolor, dVersicolor, outputWeights(2,2), aVirginica, dVirginica, outputWeights(2,3),outH2, hiddenWeights(2,1), currentInputs(1,1)); 
    hiddenWeights(2,2) = adjust1stLayer(aSetosa, dSetosa, outputWeights(2,1), aVersicolor, dVersicolor, outputWeights(2,2), aVirginica, dVirginica, outputWeights(2,3),outH2, hiddenWeights(2,2), currentInputs(2,1)); 
    hiddenWeights(2,3) = adjust1stLayer(aSetosa, dSetosa, outputWeights(2,1), aVersicolor, dVersicolor, outputWeights(2,2), aVirginica, dVirginica, outputWeights(2,3),outH2, hiddenWeights(2,3), currentInputs(3,1)); 
    hiddenWeights(2,4) = adjust1stLayer(aSetosa, dSetosa, outputWeights(2,1), aVersicolor, dVersicolor, outputWeights(2,2), aVirginica, dVirginica, outputWeights(2,3),outH2, hiddenWeights(2,4), currentInputs(4,1)); 
    
    % Adjust the weights that feed INTO Hidden Layer Neuron #3
    hiddenWeights(3,1) = adjust1stLayer(aSetosa, dSetosa, outputWeights(3,1), aVersicolor, dVersicolor, outputWeights(3,2), aVirginica, dVirginica, outputWeights(3,3),outH3, hiddenWeights(3,1), currentInputs(1,1)); 
    hiddenWeights(3,2) = adjust1stLayer(aSetosa, dSetosa, outputWeights(3,1), aVersicolor, dVersicolor, outputWeights(3,2), aVirginica, dVirginica, outputWeights(3,3),outH3, hiddenWeights(3,2), currentInputs(2,1)); 
    hiddenWeights(3,3) = adjust1stLayer(aSetosa, dSetosa, outputWeights(3,1), aVersicolor, dVersicolor, outputWeights(3,2), aVirginica, dVirginica, outputWeights(3,3),outH3, hiddenWeights(3,3), currentInputs(3,1)); 
    hiddenWeights(3,4) = adjust1stLayer(aSetosa, dSetosa, outputWeights(3,1), aVersicolor, dVersicolor, outputWeights(3,2), aVirginica, dVirginica, outputWeights(3,3),outH3, hiddenWeights(3,4), currentInputs(4,1)); 
    
end

% Displays the errors of the ith run 
disp(i + "th run errors: ");
disp("The total error is: ");
disp(errorTotal(outOutput(1,1), dSetosa, outOutput(2,1), dVersicolor, outOutput(3,1), dVirginica));
       
end

% Displays the final weight values after the program has run 
disp("Final hidden weights: ");
disp(hiddenWeights); 

disp("Final output weights: ");
disp(outputWeights);

end

% Reassign the hidden weights and output weights to testing weights 
testingHiddenWeights = hiddenWeights; 
testingOutputWeights = outputWeights; 

%%%% END TRAINING PHASE %%%%

if(testing == true)
    
    % A constant to track how many flowers were correctly identified 
    numRight = 0; 
    
    % Loop through the test data 
    for testIteration = 1:30
        testRowData = testingData(testIteration, 1:4)';
        testRowOutput = testingData(testIteration, 5:7)';
      
        % Here we do the calculations to get the inputs and outputs to all the
        % neurons in the hidden layer 
        testNetHidden = testingHiddenWeights*testRowData - biasHidden; 
        testOutHidden = arrayfun(@sigmoid,testNetHidden); 

        % Here we do the calculations to get the inputs and outputs to all the
        % neurons in the output layer 
        testNetOutput = testingOutputWeights * testOutHidden - biasOutput; 
        testOutput = arrayfun(@sigmoid,testNetOutput); 

        probSetosa = testOutput(1,1); 
        probVersicolor = testOutput(2,1);
        probVirginica = testOutput(3,1); 
        actual = ""; 

        if (probSetosa > probVersicolor) && (probSetosa > probVirginica)
            if testingData(testIteration, 5) == 1
                actual = "Setosa"; 
            elseif testingData(testIteration, 6) == 1
                actual = "Versicolor"; 
            elseif testingData(testIteration, 7) == 1
                actual = "Virginica"; 
            end 
            
            disp("Predicted: Setosa, Actual: " +  actual); 
            if testingData(testIteration,5) == 1
                 numRight = numRight+1; 
            end
        elseif (probVersicolor > probSetosa) && (probVersicolor > probVirginica)
             if testingData(testIteration, 5) == 1
                actual = "Setosa"; 
            elseif testingData(testIteration, 6) == 1
                actual = "Versicolor"; 
            elseif testingData(testIteration, 7) == 1
                actual = "Virginica"; 
            end 
            disp("Predicted: Versicolor, Actual: " +  actual); 
            if testingData(testIteration,6) == 1
                 numRight = numRight+1; 
            end
        elseif (probVirginica > probSetosa) && (probVirginica > probVersicolor)
             if testingData(testIteration, 5) == 1
                actual = "Setosa"; 
            elseif testingData(testIteration, 6) == 1
                actual = "Versicolor"; 
            elseif testingData(testIteration, 7) == 1
                actual = "Virginica"; 
            end 
            disp("Predicted: Virginica, Actual: " +  actual); 
            if testingData(testIteration,7) == 1
                 numRight = numRight+1;  
            end
        end
    
    end
    disp("The number of correct matches is: " + numRight); 
    disp("The accuracy rate is: " + numRight/30*100 + "%"); 
end

% Functions: 

function s = sigmoid(x)
    s = 1.0 / (1.0 + exp(-1.0 * x)); 
end

% A function to compute the real new weight 
function newWeight1 = adjust1stLayer(aSetosa, dSetosa, w1, aVersicolor, dVersicolor, w2, aVirginica, dVirginica, w3, outH, wToAdjust, input)

% A learning constant, n 
    n = 0.01;
    
    deltaSetosa = (aSetosa * (1.0 - aSetosa)) * (dSetosa - aSetosa);
    deltaVersicolor = (aVersicolor * (1.0 - aVersicolor)) * (dVersicolor - aVersicolor);
    deltaVirginica = (aVirginica  * (1.0 - aVirginica)) * (dVirginica - aVirginica );
    
    gradient = (deltaSetosa * w1 + deltaVersicolor * w2 + deltaVirginica * w3) * outH * (1.0-outH);
    
    deltaW = n * input * gradient; 
    
    newWeight1 = wToAdjust + deltaW; 

end

% Second layer adjustment 
function newWeight2 = adjust2ndLayer(oldW, desiredOut, actualOut, actualOutHidden)
  
    % A learning constant, a
    a = 0.001;
    eTotPartial = desiredOut - actualOut;
    outPartial = actualOut * (1.0 - actualOut);
    gradient = eTotPartial * outPartial;

    newWeight2 =  oldW + (a * actualOutHidden * gradient);

end

% Error
function e = errorTotal(aSetosa, dSetosa, aVersicolor, dVersicolor, aVirginica, dVirginica)
  
    e = 0.5 * (dSetosa - aSetosa)^2 + 0.5 * (dVersicolor - aVersicolor)^2 + 0.5 * (dVirginica-aVirginica)^2;
    
end
    
      

