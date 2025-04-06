
*Disregard this README, It is my obsidian note for this project going through the motions of building a NN using raw C++*

---

> [!info]
> - This is a note detailing implementation by providing explanations of various functions 


---

# Absolute Basics of Implementation

> [!info]
> - This section may be quite all over the place
> - It is just some design choices of note that may help during development


### Design Choices

1. **MLP as a Class**
    - Seem logical to have some class called `MLP` to hold all the functions and hyperparams

2. **Usage of Vectors**
    - I originally prioritized efficiency, thus I thought arrays everywhere was best
    - After some complexity overheads due to naivety of implementation I decided to use mostly Vectors
    - This way i can dynamically add rows, cols to my matrices
    - I may come back later to convert things to arrays where applicable
3. **Usage of Enums**
    - Stored all my hyperparam settings as Enums to give semantic meaning instead of just arbitrary numbers
    - Explanation of the different Enums can be found at: [[#Enums]]

4. **`alias.h`:**
    - Best thing I did
    - Instead of typing this like `std::vector<std::vector<std::vector<double>>>` I can now simply type `DoubleVector3D`

5. **Data Structure:**
    - Rows $\equiv$ Attributes
    - Cols $\equiv$ Examples

6. **Weights Structure:**
    - As discussed above we are storing our weights & biases in vectors
    - Our weights and biases for each layer will be bundled in a single double matrix
    - We will then have `L` matrices for `L` layers
    - This means we will store our weights & biases in a 3D Double Vector
        - ie. `DoubleVector3D`
    - **Dimensions:**
        - $(Layer, Rows, Cols)$
    - **Note:**
        - Only layers for the processing layers are saved
        - More clarification on this can be found at: [[MLP-Notation]] & [[MLP-Forward]]

# Enums

> [!info]
> - Class Enums are used to specify various hyper-parameters of the MLP

### `InitMethod`

- Used to specify the weight/bias initialisation method
- Right now we have 
	1. Uniform
	2. Gaussian

### `ActFunc`

- Used to specify the activation function used for hidden layers and output layers independently
- Right now we have 
	1. Sigmoid
	2. Tanh
	3. ReLu
	4. ELU

### `LossFunc`

- Used to specify the Loss function used
- Right now we have 
	1. SSE
	2. ENTROPY


---

# `calcLoss` Template Method

- We take in 2 numeric matrices 
	- `groundTruth`, where $\text{Rows} \equiv \text{Output Neurons}$
	- `preds`, $\text{Cols} \equiv \text{Examples}$
- Calculate loss for each instance and each output neuron and returns a matrix
- Example using SSE
	- Let
		- $Y \equiv \text{groundTruth}$
		- $\hat{Y} \equiv \text{preds}$
		- $L \equiv \text{Output}$
			- $\text{Rows} \equiv \text{Output Neurons}$
			- $\text{Cols} \equiv \text{Examples}$
	- Then the function calculates
		- $L = \frac{1}{2}(Y - \hat{Y})^2$
- With this we can aggregate over examples or output neurons if we want to 

---

# `avgLossGradient` Template Method

- We take in 2 numeric matrices 
	- `groundTruth`
	- `preds`
		- $\text{Rows} \equiv \text{Output Neurons}$
		- $\text{Cols} \equiv \text{Examples}$
- Calculate gradient of the loss for each instance and each output neuron then averages over columns with the number of examples
- Example using SSE
	- Let
		- $Y \equiv \text{groundTruth}$
		- $\hat{Y} \equiv \text{preds}$
		- $B \equiv \text{Number of Examples}$
		- $\partial L \equiv \text{Output}$
			- $\text{Rows} \equiv \text{Output Neurons}$
			- $\text{Cols} \equiv \text{Examples}$
	- Then the function calculates
		- $\partial L = \frac{1}{B}(\hat{Y} - Y)$
- Note we are still returning the same size matrix as `groundTruth` and `preds` except we have aggregated over the rows
	- (left to right)

---

# MNIST Dataset

> [!info]
> - Data set was downloaded from: [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download&select=t10k-labels.idx1-ubyte)
> - This will detail how to get this dataset and use it in the project

### What Is The `idx3-ubyte` Extension

> [!info]
> - The file type with the extension `.idx3-ubyte` is part of the `IDX` file format, which is specifically designed for storing data in the MNIST dataset (and similar datasets). 
> - This format is used to store multi-dimensional arrays of data in a compact binary format.

- **Breakdown:**
    - **IDX**: Indicates the file format, which is a simple binary format for storing arrays.
    - **3**: The number "3" specifies that the data in this file is a 3-dimensional array (e.g., images that have width, height, and number of examples).
    - **ubyte**: Stands for "unsigned byte," meaning the data elements are stored as unsigned 8-bit integers (ranging from 0 to 255).

- **For Our Case:**
    1. **Images** (`t10k-images.idx3-ubyte`):
       - 3-dimensional arrays: `[number of images] x [rows] x [columns]`
       - Each image is represented as a 28x28 grid of grayscale pixel values (0-255).
    2. **Labels** (`t10k-labels.idx1-ubyte`):
       - 1-dimensional array: `[number of images]`
       - Each label is a single unsigned byte representing the digit (0-9).

### How to Load These Files

> [!info]
> - I refer you to: [[C++Reading-Writing#Reading Binary Files]]
> - Here we go through some nuances not discussed above

1. **Structure Of The File**
    - These binary files have a specified format discussed in: [MNIST handwritten digit database, Yann LeCun, Corinna Cortes and Chris Burges](https://yann.lecun.com/exdb/mnist/)
    - Below is a breakdown:
    - **Both `idx3` files:**
        - This is the Image Data
        - I begins with 4 32-bit integers
            1. Magic Number (can be mostly ignored)
            2. Number Of Images
            3. Number Of Rows Per Image
            4. Number Of Cols Per Image
        - These 32-bit Images follow the Big-Endian Format
            - Thus we reverse the endianess of these numbers
        - After that follows 8-bit unsigned integers 
            - These represent the pixel values (0-255)
    - **Both `idx1` files:**
        - This is the Label Data
        - I begins with 2 32-bit integers
            1. Magic Number (can be mostly ignored)
            2. Number Of Labels
        - These 32-bit Images follow the Big-Endian Format
            - Thus we reverse the endianess of these numbers
        - After that follows 8-bit unsigned integers 
            - These represent the labels (0-9)

2. **Reversing Endianess:**
    - To do this we use the function `__builtin_bswap32()`
            
3. **Use of The `.data()` Method:**
    - We know from [[C++Libraries#The `.data()` Method]] this method returns the pointer of the first element
    - We know from [[C++Reading-Writing#Reading Binary Files]] that we can populate a buffer using its pointer
    - Thus, this is what we doing
    - **Alternative**
        - Instead we could have populated each element in a loop
            ```cpp
            for (int i = 0; i < numLabels; i++) {
                finLabels.read(reinterpret_cast<char*>(&labels[i]), 1);
            }
            ```
        - This method is very inefficient 
        - It also requires a read check on every iteration using `.gcount()`
        - The preferred method only requires one `.gcount()` check at the end

4. **Structure of Output:**
    - We create a `struct` called `DataMNIST` which holds 2 values:
        1. `Uint8Vector3D imgs;`
        2. `std::vector<uint8_t> labels;`
    - This `struct` is populated with our imported data and returned in the function
   
# Saving MLP Model

> [!info]
> - I am saving these models as binary data
>   - This means saving it as a `.bin` file
>   - I am choosing `.bin` instead of something more descriptive like `idx3-ubyte` because this is for my use case
>       - If I was making something for production or whatever then I would use a more descriptive extension
> - These models are stored in the `models/` directory
> - Specification of the models will be stored as headers in the binary file

- **General Implementation:**
    - We start with headers then the bulk of the file will be weight and bias values
    - Will store values in little-endian format
        - This takes away complexity when reading the file
        - Furthermore, chat told me most modern systems prefer it

### Headers

> [!info]
> - Storing an MLP means storing not only the weights and biases, but also the various hyperparams
> - This information is going to be stored as headers in our binary file
> - Here we specify the structure of these headers
> - I will follow the format give in: [MNIST handwritten digit database, Yann LeCun, Corinna Cortes and Chris Burges](https://yann.lecun.com/exdb/mnist/)


- **Things To Store:**
    1. Hidden Layer Activation Function (Enum)
    2. Output Layer Activation Function (Enum)
    3. Loss Function (Enum)
    4. Initial Learning Rate (double)
    5. Batch Size (8-bit integer)
    6. Weights & Biases (3D Vector of doubles)

- **How to Store Enums:**
    - We store Enums as 8-bit integers, since is all they are
    - Note if we mess around with the order of our Enums in the source code then this won't work

- **How to Store Doubles:**
    - Most modern systems store doubles in 64-bits (8 bytes)
    - My system does this and can be confirmed by the code:
        ```cpp
        std::cout << "Size of a `double` on this system is: " << sizeof(double) << std::endl;
        ```
    - Thus, we will assign 64-bits to each double

- **How to Store Structure:**
    - So of course, the information of the structure is embedded in the weights & biases
    - Issue is that the structure is dynamic
    - Thus when specifying the structure we:
        1. Give the number of layers (call that `l`)
        2. Then we give the rows and cols of each matrix for the next `l` layers 
    - This implementation mean the size of the header will vary between saved models

- **Consolidation on Getting Weight Matrices:**
    - Here we discuss exactly how to get the width of each layer
    - This is mostly trivial since this doesn't affect how we populate our matrices but may be good for understanding
    - Essentially, we must remember that the rows/cols of the matrices at each layer define the width of each layer
        - Rows $\equiv$ Neurons Going To
        - Cols $\equiv$ Neurons Coming From
    - **Note:**
        - We have an additional, pre-padded Column that is reserved for the Biases
        - This makes sense since these biases aren't "going to" an other neurons
        - Thus, we have 1 more column than rows
    - For more clarification, revisit both: [[MLP-Notation]] & [[MLP-Forward]] 

- **File Format**
    - `models/test.bin`

| offset |          type          | value |               description               |
| :----: | :--------------------: | :---: | :-------------------------------------: |
|  0000  | unsigned 8-bit integer |  ??   | Hidden Layer Activation Function (Enum) |
|  0001  | unsigned 8-bit integer |  ??   | Output Layer Activation Function (Enum) |
|  0002  | unsigned 8-bit integer |  ??   |          Loss Function (Enum)           |
|  0003  |      64-bit float      |  ??   |          Initial Learning Rate          |
|  0011  | unsigned 8-bit integer |  ??   |               Batch Size                |
|  0012  | unsigned 8-bit integer |  ??   |            Number of Layers             |
|  0013  |     32-bit integer     |  ??   |     Weight Matrix Rows (1st Layer)      |
|  0017  |     32-bit integer     |  ??   |     Weight Matrix Rows (2nd Layer)      |
|  ....  |          ....          | ....  |                  ....                   |
|  xxxx  |     32-bit integer     |  ??   |     Weight Matrix Rows (Lth Layer)      |
|  xxxx  |     32-bit integer     |  ??   |     Weight Matrix Cols (1st Layer)      |
|  xxxx  |     32-bit integer     |  ??   |     Weight Matrix Cols (2nd Layer)      |
|  ....  |          ....          | ....  |                  ....                   |
|  xxxx  |     32-bit integer     |  ??   |     Weight Matrix Cols (Lth Layer)      |
|  xxxx  |      64-bit float      |  ??   |          Weight or Bias Value           |
|  ....  |          ....          | ....  |                  ....                   |


# Future Improvements

> [!info]
> - This section details of future implementations i would like to do for this project


1. **Convert Vectors To Arrays:**
    - I have been using vectors exclusively in this project since it is easier to use
    - This probably makes things slower
    - Thus, convert vectors to arrays where applicable

2. **Proper Error Handling:**
    - There is a lot of things that can go wrong with the functions if some fucky input is given
    - Thus, I would like to add some proper error handling

3. **Adam Optimizer:**
    - Want to implement this feature







