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
- Need to learn how to import csv files



### Possible improvements

- Change forward prop algo to handle bias calcs separaetey so we dont prepad the `inp` vector on each itteration
- Take note of the TODO flags all over the code

### References

- Data set is from: [Cyber Security Attacks](https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks?resource=download&select=cybersecurity_attacks.csv)

