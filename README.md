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
- Matrix multiplication is break, idk whats happening
    - I suspect incorrect indexing
