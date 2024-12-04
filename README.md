# RNN

# Proccess:

- I want to implement an RNN in C++ using classes
- I am going to first implement an MLP since i know how to do that

### MLP proccess

- Fell at the first hurdle of creating the class
- Wanted to use arrays to manage the structure of the mlp
- This wont work since the mlp can vary in depth and width
- Thus, I need to learn Vectors
    - Back to C-PlusPlus repo
- Im back
- in our Mlp class, we have a vector containing weight matrices which are also vectors
- Right now, weigth init is between -1 and 1, might change weight initialisation to a separate method that allows for different init methods
    - Used enums to specify the method. must figure out how to handle different parameters for different methods
        - Maybe just pass in a vector or array or whatever
- Implemented Sigmoid and tanh activation functions
- Going to do a forward prop thingy
- Matrix mult working
- Start thinking about error handling within classes and things like this idk
    - Maybe build my sustem then i can think of errors when im done
    - Learn how error handling is done is prolly the best idea before i start doing it
- Developing Forward prop
    - Debug the function, its not working. 
    - I think it has to do with appending the inp vector
    - Fix it
- Forward Prop working
    - The problem was that we were redclaring the inp variable when it was returned (silly)
    - Note that I may want to try a different implementation of handling bias separaetely to avoid having to pre pad our `inp` vector on each loop
    - Right now it just performs tanh everywhere
        - This means including the output

### Backprop

- Going to make a function for a single wieght update itteration
- Then another function to do BGD

#### Single iteration developement plan

- take in input batch
- run forward pass
- We then get the average loss of the batch
    - ie. calc `pred - actual` for each instance then take the average
- calc differential wrt each neuron using this average loss value
- then with these neuron differentials we can calc weight updates very easily
- update weights in this single iteration function

#### BGD developement plan

- Split data accordingly into batches
- perform iteration in each batch
- do this for all batches for x amount of epochs

#### Hyperparams

- Handle the following hyperparams by making them member variables
    - Learning rate (and exp decay)
    - Loss function
    - Batch Size

#### Considerations

- Normailisation of inputs?
    - preproccessing step, can do this as a separate function
- For each forward pass, ensure we store intermediate results
    - TODO: think more about this
- Shuffle these batches at the beginning of each epoch
    - Handle datasets that can't be split perfectly into x amount of batches
- Track loss function at each iteration
- Implement early stopping

### Back to process

- Forward prop revisited
    - Need to store intermediate results for back prop
        - output of net function (z) and the output of the activations (a)
    - This means instead of right now just returning the output of the forward pass we need to return 2 3d vectors
        - Why 3D vector?
            - understand that we are performing a single forward pass on a batch ie. mutliple instances
            - this the dimensions will be (layer, neuron, instance)
    - Storing options
        - Chat recommends this `std::pair` from the `<utility>` library
            - Going to go learn that, I'll be back eventually
- I'm back
    - Can now store intermediate values
    - We are using a struct to store the result
        - This is defined in `mlp.h`
        - Note the behaviour:
            - Last element of z is the same as the first element of a
    - Also implemented alias that are in alias.h

- Now doing Getters and setters then going to testing to ensure they work an constructor as well
- When you see this again, do the printing of the enums in the helper functions
- All getters and setters work
- Loss function done
    - Maybe add template to hadnle both types, I dont want to rewrite all the functions code
    - Cant do entropy for doubles though???
    - Figure this shit out
- change result param from calc loss function to predictions rather
- functions to calcluate errors and average loss implemented
    - Right now they only do MSE
    - They take in a matrix of ground truths and predictions (rows $\equiv$ output neurons, cols $\equiv$ examples)
    - This means our output is a 1D of size equal to the number of ourput neurons
    - Similar for avg loss for neuron differentials
- Need to learn how to import csv files
- ffs learn file handling

- Im back after 3 days, I undertsand how to do things but will do later
- With the current functionallity, We cna import a csv into a 2d vector of strings
- The idea is that we can now take in any csv and we need another function to process each element into the types we want
- There is a problem
    - Right now, We are splitting each row by ','
    - This is bad since some string elements that are wrapped in "" has ',' in them
    - Need to ignore these
    - Fix
        - Have a separate flag to determine of the next element is in quotes
        - If not then normal things
        - If there are then handle it

- Ok, I got things working, but it can't handle elements that have embedded quotes
    - ie. an element that looks like this won't work: `12,14,3.2,"This is a ""Quoted"" Element",76,10`
- New lines are fucking with me :D
- I may be done with this shit
- idk, maybe I'll get a burst of motivation to do this
- Ok, im tackling it again
- Plan
    - Read from the csv until we get a complete record
    - Function will look really ugly
    - Idea is to read the column headings, count them and process x amount of elements before appending a row
- Ok fine, I concede, I will use an importing library
- My function only barely works, and its extrememly slow
- Decided to use a simple dataset with no weird edge cases
- What Dataset looks like 
    - rowIdx
    - earnings
    - Education Level: [ "bachelor", "highschool" ] 
    - Gender: [ "male", "female" ]
    - Age
- What my `Data` looks like
    - bachelor [ 0, 1 ]
    - highschool [ 0, 1]
    - male [ 0, 1]
    - female [ 0, 1]
    - age (z-score normalised)
    - earnings (target)

- Ok, everything works
- Now time for backprop
- Added functionality to transpose input for forward prop and now things should work fine
- BackProp For real now
- I learnt fully how backprop works now
    - This required a small change to forward prop
    - We now save our itermediate outputs with an appended row of 1s for hidden layers
        - That is z and a in hidden layers have pre-padded row of 1s
        - Whereas z and a in output layer does not have pre-padded row od 1s
- Ok, updating avgLoss template function, do that first PLEASE

- Loss functions and avg gradient functions i am now happy with. 
- They take in matrices for targets so i need to change my separateTarget() function to handle this
    - Maybe specify the columns where the target lives to create new matrix

- `separateTarget()` works with some tech debt
    - The order of the targets is ignored in the outputted matrix
    - Instead the indicies are ordered in descending order
    - So if our targetCols are [3, 6, 1] then, the output matrix will be [6, 3, 1]

- Ok, Changed dimensions of data
- Need to make functions for activations and their gradients

- THINK ABOUT
    - Does the 1s get gradient function treatment? does it matter with the current acti funcs?
        - As in maybe an input of 1 returns a 1?
        - Answer is no
    - I wanna eat now so not doing this now
    - Final, the pre-padded 1s should not be run through the gradient activation function so changing to that
    - this might be the cause of some bugs so come back here if something goes wrong

- I think it worked
- Things changed
    - Forward prop result returns no prepadding for Z
    - But pre-padded row of 1s for A's that are not output layer
- Backprop
    - See Obsidian notes
    - But some things of note
        - Weights are spearated from bias column before calcs
        - A's are prepadded from forward prop
        - input is also prepadded since this is an A matrix in the 0th layer (not output layer)

### Things to note as you develop Coeni

- `data`
    - Rows = Attributes 
    - Columns = rows
    - Note that this was a late decision, so there is some weird initialisation stuffs with transposing
- `separateTarget()`
    - Note the order of columns (line 165)
- `calcLoss()`
    - Returns a matrix such that we can perform aggregations how we like
    - Inputs reuquire (Rows = output neurons, Cols = examples)
- `elu()` activation function is using a backed in aplha value of 1
    - This was done to make function pointers simpler
- Default `MLP` constructor
    - HiddenLayerActFunc = RELU
    - OutputLayerActFunc = SIGMOID
    - Loss Function = MSE
    - LR = 0.1
    - DecayRate = 0
    - BatchSize = 32
- Both `applyActivation()` and `applyGradientAct()` alters the input matrix
    - This could be the root of some bugs so just note this

### Possible improvements

- Change forward prop algo to handle bias calcs separaetey so we dont prepad the `inp` vector on each itteration
- Take note of the TODO flags all over the code

### References

- Data set is from: [Cyber Security Attacks](https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks?resource=download&select=cybersecurity_attacks.csv)
- Library used to import csv files: [ben-strasser/fast-cpp-csv-parser: fast-cpp-csv-parser](https://github.com/ben-strasser/fast-cpp-csv-parser) 
- New DataSet: [vincentarelbundock.github.io/Rdatasets/datasets.html](https://vincentarelbundock.github.io/Rdatasets/datasets.html)
    - CPSSW04.csv

