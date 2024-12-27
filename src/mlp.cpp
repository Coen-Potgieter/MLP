#include "mlp.h"
#include "helperfuncs.h"
#include <cstdint>


MLP::MLP(const std::vector<int>& inpStructure, const ActFunc inpHiddenLayerAct, const ActFunc inpOutputLayerAct, const LossFunc inpLossFunc, const float inpLR, const int inpDecayRate, const int inpBatchSize) {

    // Populate weights
    for (size_t layerIdx = 0; layerIdx < inpStructure.size() - 1; layerIdx++) {
        const int cols = inpStructure[layerIdx];
        const int rows = inpStructure[layerIdx + 1];

        // Construct the empty weight matrix (cols plus 1 for each bias)
        std::vector<std::vector<double>> weightMatrix(rows, std::vector<double>(cols + 1, 0.0));

        // Now assign wieght matrix to out member variable
        this->weights.push_back(weightMatrix);
    }
    // Populate other member variables
    this->hiddenLayerAct = inpHiddenLayerAct;
    this->outputLayerAct = inpOutputLayerAct;
    this->lossFunc = inpLossFunc;
    this->initialLR = inpLR;
    this->decayRate = inpDecayRate;
    this->batchSize = inpBatchSize;
}
MLP::~MLP() = default;

// Getters
DoubleVector3D MLP::getWeights() const {
    return this->weights;
}
MLP::ActFunc MLP::getHiddenLayerAct() const {
    return this->hiddenLayerAct;
}
MLP::ActFunc MLP::getOutputLayerAct() const {
    return this->outputLayerAct;
}
MLP::LossFunc MLP::getLossFunc() const {
    return this->lossFunc;
}
float MLP::getLR() const {
    return this->initialLR;
}
int MLP::getDecayRate() const {
    return this->decayRate;
}
int MLP::getBatchSize() const {
    return this->batchSize;
}
// Setters
void MLP::setWeights(const DoubleVector3D& inpWeights) {
    // First check for valid dimensions
    const uint8_t inpNumLayers = inpWeights.size();
    // Layers check
    if (inpNumLayers != this->weights.size()) {
        std::cerr << "Input Wieghts Has Invalid Number of Layers" << std::endl;
        return;
    }

    // Rows & Cols check at each layer
    size_t inpNumRows;
    size_t inpNumCols;
    for (size_t layerIdx = 0; layerIdx < inpNumLayers; layerIdx++) {
        inpNumRows = inpWeights[layerIdx].size();
        if (inpNumRows != this->weights[layerIdx].size()) {
            std::cerr << "Input Weights Has Invalid Number of Rows for Layer: " << layerIdx << std::endl;
            return;
        }
        inpNumCols = inpWeights[layerIdx][0].size();
        if (inpNumCols != this->weights[layerIdx][0].size()) {
            std::cerr << "Input Weights Has Invalid Number of Cols for Layer: " << layerIdx << std::endl;
            return;
        }
    }

    // Now simple assignment
    for (size_t layerIdx = 0; layerIdx < inpNumLayers; layerIdx++) {
        this->weights[layerIdx] = inpWeights[layerIdx];
    }
    return;

}
void MLP::setHiddenLayerAct(const ActFunc newAct) {
    this->hiddenLayerAct = newAct;
}
void MLP::setOutputLayerAct(const ActFunc newAct){
    this->outputLayerAct = newAct;
}
void MLP::setLossFunc(const LossFunc newLossFunc) {
    this->lossFunc = newLossFunc;
}
void MLP::setLR(const float newLR) {
    this->initialLR = newLR;
}
void MLP::setDecayRate(const int newDecayRate) {
    this->decayRate = newDecayRate;
}
void MLP::setBatchSize(const int newBatchSize) {
    this->batchSize = newBatchSize;
}

void MLP::initWeights(InitMethod method, const double minVal, const double maxVal) {

    switch(method) {
        case MLP::InitMethod::UNIFORM_RANDOM:
            initUniformRandom(this->weights, minVal, maxVal, true);
            break;
        case MLP::InitMethod::GAUSSIAN_RANDOM:
            break;
        default:
            std::cerr << "This Init Method Has Not Yet Been Implemented..." << std::endl;
            break;
    }
}

void MLP::initBias(InitMethod method, const double minVal, const double maxVal) {
    switch(method) {
        case MLP::InitMethod::UNIFORM_RANDOM:
            initUniformRandom(this->weights, minVal, maxVal, false);
            break;
        case MLP::InitMethod::GAUSSIAN_RANDOM:
            break;
        default:
            std::cerr << "This Init Method Has Not Yet Been Implemented..." << std::endl;
            break;
    }
}

void MLP::applySoftmax(DoubleVector2D& Z) const {

    const size_t numRows = Z.size();
    const size_t numCols = Z[0].size();
    // Loop through Cols
    std::vector<double> singleCol(numRows);
    for (size_t colIdx = 0; colIdx < numCols; colIdx++){
        // Gather Col Vals
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            singleCol[rowIdx] = Z[rowIdx][colIdx];
        }
        // Apply Softmax
        singleCol = softmaxHandler(singleCol);

        // Write softmaxxed vals back to matrix
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            Z[rowIdx][colIdx] = singleCol[rowIdx];
        }
    }
    return;
}

void MLP::applyActivation(DoubleVector2D& Z, const size_t layer) const {

    // Ensure `Z` is not empty
    if (Z.empty()) {
        throw std::invalid_argument("`Z` is empty");
        return;
    }

    const size_t numLayers = this->weights.size();
    // Ensure valid `layer` argument
    if (layer >= numLayers) {
        throw std::invalid_argument("`layer` is out of valid layer range");
        return;
    }

    // Based on `layer` choose which act func to use
    MLP::ActFunc actFuncToUse;
    if (layer == (numLayers -1)) {
        actFuncToUse = this->getOutputLayerAct();
    } else {
        actFuncToUse = this->getHiddenLayerAct();
    }

    // Set up function pointer
    double (*actFuncPtr)(const double);
    switch(actFuncToUse) {
        case MLP::ActFunc::SIGMOID:
            actFuncPtr = &sigmoid;
            break;
        case MLP::ActFunc::TANH:
            actFuncPtr = &tanh;
            break;
        case MLP::ActFunc::RELU:
            actFuncPtr = &relu;
            break;
        case MLP::ActFunc::ELU:
            actFuncPtr = &elu;
            break;
        case MLP::ActFunc::SOFTMAX:
            return applySoftmax(Z);
            break;
        default:
            throw std::logic_error("Activation Function Has Not Yet Been Implemented");
    }



    // Apply Activation Function to each element
    for (std::vector<double>& row : Z) {
        for (double& elem : row) {
            elem = actFuncPtr(elem);
        }
    }

    return;
}
void MLP::applyGradientActivation(DoubleVector2D& Z, const size_t layer) const {

    // Ensure `Z` is not empty
    if (Z.empty()) {
        throw std::invalid_argument("`Z` is empty");
        return;
    }

    const size_t numLayers = this->weights.size();
    // Ensure valid `layer` argument
    if (layer >= numLayers) {
        throw std::invalid_argument("`layer` is out of valid layer range");
        return;
    }

    // Based on `layer` choose which act func to use
    MLP::ActFunc actFuncToUse;
    if (layer == (numLayers -1)) {
        actFuncToUse = this->getOutputLayerAct();
    } else {
        actFuncToUse = this->getHiddenLayerAct();
    }

    // Set up function pointer
    double (*actFuncPtr)(const double);
    switch(actFuncToUse) {
        case MLP::ActFunc::SIGMOID:
            actFuncPtr = &derivativeSigmoid;
            break;
        case MLP::ActFunc::TANH:
            actFuncPtr = &derivativeTanh;
            break;
        case MLP::ActFunc::RELU:
            actFuncPtr = &derivativeRelu;
            break;
        case MLP::ActFunc::ELU:
            actFuncPtr = &derivativeElu;
            break;
        default:
            throw std::logic_error("Activation Function Has Not Yet Been Implemented");
    }


    // Apply Activation Gradient to each element
    for (std::vector<double>& row : Z) {
        for (double& elem : row) {
            elem = actFuncPtr(elem);
        }
    }

    return;
}

// Version 1, might change this to handle biases differently instead of always appending a vector of 1s (this may be slower)
ForwardPropResult MLP::forwardProp(const DoubleVector2D& inpQuery) const {

    DEBUG_LOG("Performing Forward Prop");

    // Declare varables to store initermediate values (See README why its 3D)
    DoubleVector3D z; 
    DoubleVector3D a; 

    // Make a copy of our input
    DoubleVector2D inp = inpQuery;

    // Row of 1s delcared outside of the for loop since row it is consistent for all iterations
    std::vector<double> row1s(inp[0].size(), 1.0);
    const size_t numLayers = this->weights.size();

    // Pred-pad our input with a row of 1s
    inp.insert(inp.begin(), row1s);

    // Perform forward prop
    for (int layer = 0; layer < numLayers; layer++){


        // For Debugging
        DEBUG_LOG("`weights` Matrix * `inp` Matrix: " 
                  << this->weights[layer].size() << "x" << this->weights[layer][0].size()
                  << " * " << inp.size() << "x" << inp[0].size());

        // wieght matrix multipled with our inp and store result
        inp = matrixMultiply(this->weights[layer], inp);

        z.push_back(inp);

        // apply activation function and store result
        applyActivation(inp, layer);

        // Pred-pad our input with a row of 1s
        if (layer != (numLayers - 1)){
            inp.insert(inp.begin(), row1s);
        }
        a.push_back(inp);
    }

    // Create an populate struct
    ForwardPropResult results;
    results.z = z;
    results.a = a;
    return results;
}

// See Obsidian Notes
void MLP::singleBackPropItter(const DoubleVector2D& inpBatch, const DoubleVector2D& target) {

    DEBUG_LOG("Performing BackProp");
    size_t numLayers = this->weights.size();
    const size_t numExamples = inpBatch[0].size();

    // Create 3D matrcies to store nueron differentials, weight updates
    DoubleVector3D weightUpdates(numLayers);
    DoubleVector3D layerDifferentials(numLayers);
    // To Store matrices depending on layer
    DoubleVector2D tmpLayerDifferentials;

    // Forward Prop run
    ForwardPropResult resForwardProp = this->forwardProp(inpBatch);

    // Separate Weights from Biases
    DoubleVector3D onlyWeights(numLayers);

    // PrePad our input with layer of 1s for later use
    DoubleVector2D inp = inpBatch;
    std::vector<double> row1s(inp[0].size(), 1.0);
    inp.insert(inp.begin(), row1s);

    bool softmaxMode = (this->getOutputLayerAct() == MLP::ActFunc::SOFTMAX); 

    for (int layer = numLayers - 1; layer >= 0; layer--) {

        DEBUG_LOG("BackProp for Layer: " << layer);
        // Apply activation Gradient to Net Function output Z
        if (!(softmaxMode && (layer == (numLayers - 1)))) {
            applyGradientActivation(resForwardProp.z[layer], layer);
        }

        // Get Current Weights (slice out first col)
        onlyWeights[layer] = sliceCols(this->weights[layer], 1, this->weights[layer][0].size() -1);

        if (layer == (numLayers - 1)){
            // If Output Layer get Loss Gradient
            tmpLayerDifferentials = this->lossGradient(target, resForwardProp.a[layer]);
        } else {
            // If Hidden Layer then use nueron differential matrix one layer up
            tmpLayerDifferentials = matrixMultiply(transpose(onlyWeights[layer + 1]), layerDifferentials[layer+1]);
        }

        // For Softmax & CE (expirimental)
        if (softmaxMode && (layer == (numLayers - 1))) {
            layerDifferentials[layer] = tmpLayerDifferentials;
        } else {
            layerDifferentials[layer] = elementWiseMatrixMultiply(tmpLayerDifferentials, resForwardProp.z[layer]);
        }


        // Neuron Differentials Done, now we can do weight and bias updates
        if (layer == 0) {
            // If first layer we need to use inputs or A^(0)
            weightUpdates[layer] = matrixMultiply(layerDifferentials[layer], transpose(inp));
        } else {
            weightUpdates[layer] = matrixMultiply(layerDifferentials[layer], transpose(resForwardProp.a[layer-1]));
        }
    }

    double update;
    // Now Update the weights
    for (size_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        for (size_t rowIdx = 0; rowIdx < this->weights[layerIdx].size(); rowIdx++) {
            for (size_t colIdx = 0; colIdx < this->weights[layerIdx][rowIdx].size(); colIdx++){

                // Do Batch Avgeraging Here
                update = weightUpdates[layerIdx][rowIdx][colIdx] / numExamples;
                this->weights[layerIdx][rowIdx][colIdx] -=  this->initialLR * update;
            }
        }
    }
    return;
}

void MLP::miniBatchGD(const DoubleVector2D& data, const DoubleVector2D& target, const size_t numEpochs) {

    // Check if Softmax is not paired with CE loss
    bool softmaxApplied = (this->getOutputLayerAct() == MLP::ActFunc::SOFTMAX);
    bool entropyApplied = (this->getLossFunc() == MLP::LossFunc::ENTROPY);

    // Logical XOR
    if (!softmaxApplied != !entropyApplied){
        std::cerr << "Must have either Both Softmax & Entropy or Neither, Stop trying to fuck my system :/" << std::endl;
        return;
    } 

    const size_t numExamples = data[0].size();
    const size_t numLayers = this->weights.size();

    // Ensure Number of examples in data matches with target
    if (target[0].size() != numExamples) {
        throw std::invalid_argument("Number of Examples in `data` does not match `target`");
    }

    // 1. Split the Data
    const int numWholeBatches = numExamples / this->batchSize;
    const int lastBatchSize = numExamples % this->batchSize;
    size_t batchStartIdx;
    size_t batchEndIdx;
    DoubleVector2D dataBatch;
    DoubleVector2D targetBatch;
    DoubleVector2D lossMatrix;
    ForwardPropResult forwardRes;
    double totalError;

    // 2. Run backprop on each batch for x epochs
    for (size_t epochIdx = 0; epochIdx < numEpochs; epochIdx++){
        for (size_t batchIdx = 0; batchIdx < numWholeBatches; batchIdx++) {
            // Get Batch
            batchStartIdx = batchIdx * this->batchSize;
            batchEndIdx = batchStartIdx + this->batchSize - 1;
            dataBatch = sliceCols(data, batchStartIdx, batchEndIdx);
            targetBatch = sliceCols(target, batchStartIdx, batchEndIdx);

            this->singleBackPropItter(dataBatch, targetBatch);
        }
        // Remaining examples that dont fit neatly into batches
        if (lastBatchSize > 0) {
            dataBatch = sliceCols(data, batchEndIdx + 1, numExamples - 1);
            targetBatch = sliceCols(target, batchEndIdx + 1, numExamples - 1);
            this->singleBackPropItter(dataBatch, targetBatch);
        }


        // 3. Assess Error
        forwardRes = this->forwardProp(data);
        totalError = calcLoss(target, forwardRes.a[numLayers-1]);

        // Check if nan
        if (totalError != totalError) {
            std::cerr << "Nan Occurred" << std::endl;
            DoubleVector3D nanWeights = this->getWeights();
            for (const DoubleVector2D& layer : nanWeights) {
                printMatrix(layer);
            }
            return;
        }
        std::cout << "Total Error For Epoch " << epochIdx + 1 << ": " << totalError << std::endl;
    }
}

void MLP::saveModel(std::string_view filePath) const {

    // Check if File Exists
    fs::path fp = filePath;

    // Playing with User input here to either overwite existing file or not
    std::string choice = "y";
    if (fs::exists(fp)) {
        bool validInput = false;
        std::cout << "This File Already Exists, Would you like to overwrite?\n(y/n): ";
        while (!validInput){
            std::cin >> choice;
            if (choice == "y" || choice == "Y" || choice == "n" || choice == "N") {
                validInput = true;
            } else {
                std::cout << "Incorrect Input\n Type Either (y/n): ";
            }
        }
        if (choice == "n" || choice == "N") {
            std::cout << "Not Saving Model" << std::endl;
            return;
        } 
    } else {
        // Create Directory If it doesnt exist
        fs::path parentPath = fp.parent_path();
        fs::create_directories(parentPath);
        DEBUG_LOG("Created Directory: " << parentPath);
    }

    // Prepare data to store
    uint8_t hiddenActFunc = static_cast<uint8_t>(this->getHiddenLayerAct());
    uint8_t outputActFunc = static_cast<uint8_t>(this->getOutputLayerAct());
    uint8_t lossFunc = static_cast<uint8_t>(this->getLossFunc());
    double initialLR = this->getLR();
    uint8_t batchSize = static_cast<uint8_t>(this->getBatchSize());
    // Prepare Weight&Bias header data
    DoubleVector3D weights = this->getWeights();
    uint8_t numLayers = static_cast<uint8_t>(weights.size());
    int32_t numRows;
    int32_t numCols;

    std::ofstream fout;
    fout.open(filePath, std::ios::binary);
    if (!fout) {
        std::cerr << "Error Writing to File: " << filePath << std::endl;
    }

    fout.write(reinterpret_cast<const char*>(&hiddenActFunc), 1);
    fout.write(reinterpret_cast<const char*>(&outputActFunc), 1);
    fout.write(reinterpret_cast<const char*>(&lossFunc), 1);
    fout.write(reinterpret_cast<const char*>(&initialLR), sizeof(double));
    fout.write(reinterpret_cast<const char*>(&batchSize), 1);
    fout.write(reinterpret_cast<const char*>(&numLayers), 1);
    for (size_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        numRows = static_cast<int32_t>(weights[layerIdx].size());
        fout.write(reinterpret_cast<const char*>(&numRows), sizeof(int32_t));
    }
    for (size_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        numCols = static_cast<int32_t>(weights[layerIdx][0].size());
        fout.write(reinterpret_cast<const char*>(&numCols), sizeof(int32_t));
    }

    if (!fout.good()) {
        std::cerr << "Error Writing Headers to the File" << std::endl;
        fout.close();
        return;
    } 

    int varNumCols;
    for (size_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        varNumCols = weights[layerIdx][0].size();
        for (size_t rowIdx = 0; rowIdx < weights[layerIdx].size(); rowIdx++){
            fout.write(reinterpret_cast<const char*>(weights[layerIdx][rowIdx].data()), varNumCols * sizeof(double));
        }
    }

    if (!fout.good()) {
        std::cout << "Error Writing Weights & Biases to the File" << std::endl;
        fout.close();
        return;
    } else {
        std::cout << "Successfully Wrote to: " << filePath << std::endl;
    }

    fout.close();
    return;
}

void MLP::loadModel(std::string_view filePath) {

    // Check if Path esists And if it is .bin file
    fs::path fp = filePath;

    std::cout << "Loading Model..." << std::endl;
    if (!fs::exists(fp)) {
        std::cerr << "Given File Path: " << filePath << " Does not exist" << std::endl;
        return;
    } else if (fp.extension() != ".bin") {
        std::cerr << "Given File Path: " << filePath << " has a '"<< fp.extension()
            << "' extension, It must have a \".bin\" extension" << std::endl;
        return;
    }

    // Prepare variables to read into
    uint8_t readHiddenActFunc;
    uint8_t readOutputActFunc;
    uint8_t readLossFunc; 
    double readInitialLR;
    uint8_t readBatchSize; 
    uint8_t readNumLayers = 0; 

    std::ifstream fin;
    fin.open(filePath, std::ios::binary);
    if (!fin) {
        std::cerr << "Error Writing to File: " << filePath << std::endl;
        return;
    }

    fin.read(reinterpret_cast<char*>(&readHiddenActFunc), 1);
    if (fin.gcount() < 1) {
        std::cerr << "Error reading Hidden Layer Activation Function" << std::endl;
        fin.close();
        return;
    }
    fin.read(reinterpret_cast<char*>(&readOutputActFunc), 1);
    if (fin.gcount() < 1) {
        std::cerr << "Error reading Output Layer Activation Function" << std::endl;
        fin.close();
        return;
    }
    fin.read(reinterpret_cast<char*>(&readLossFunc), 1);
    if (fin.gcount() < 1) {
        std::cerr << "Error reading Loss Function" << std::endl;
        fin.close();
        return;
    }
    fin.read(reinterpret_cast<char*>(&readInitialLR), sizeof(double));
    if (fin.gcount() < 1) {
        std::cerr << "Error reading Initial Learning Rate" << std::endl;
        fin.close();
        return;
    }
    fin.read(reinterpret_cast<char*>(&readBatchSize), 1);
    if (fin.gcount() < 1) {
        std::cerr << "Error reading Batch Size" << std::endl;
        fin.close();
        return;
    }
    fin.read(reinterpret_cast<char*>(&readNumLayers), 1);
    if (fin.gcount() < 1) {
        std::cerr << "Error reading Number of Layers" << std::endl;
        fin.close();
        return;
    }

    std::vector<int32_t> numRows(readNumLayers);
    std::vector<int32_t> numCols(readNumLayers);
    fin.read(reinterpret_cast<char*>(numRows.data()), readNumLayers * sizeof(int32_t));
    if (fin.gcount() < readNumLayers * sizeof(int32_t)) {
        std::cerr << "Error reading Number of Rows each Layer" << std::endl;
        fin.close();
        return;
    }
    fin.read(reinterpret_cast<char*>(numCols.data()), readNumLayers * sizeof(int32_t));
    if (fin.gcount() < sizeof(int32_t)) {
        std::cerr << "Error reading Number of Cols each Layer" << std::endl;
        fin.close();
        return;
    }

    DoubleVector3D readWeights;
    for (size_t layerIdx = 0; layerIdx < readNumLayers; layerIdx++) {
        DoubleVector2D layerMatrix(numRows[layerIdx], std::vector<double>(numCols[layerIdx]));
        std::vector<double> layerRow(numCols[layerIdx]);
        for (size_t rowIdx = 0; rowIdx < numRows[layerIdx]; rowIdx++) {

            fin.read(reinterpret_cast<char*>(layerRow.data()), numCols[layerIdx] * sizeof(double));
            if (fin.gcount() < numCols[layerIdx] * sizeof(double)) {
                std::cerr << "Error Reading Row: " << rowIdx << " At Layer: " << layerIdx << std::endl;
            }
            layerMatrix[rowIdx] = layerRow;
        }
        readWeights.push_back(layerMatrix);
    }
    fin.close();

    // Now Assign new Params/Hyperparams
    this->setHiddenLayerAct(static_cast<ActFunc>(readHiddenActFunc));
    this->setOutputLayerAct(static_cast<ActFunc>(readOutputActFunc));
    this->setLossFunc(static_cast<LossFunc>(readLossFunc));
    this->setLR(readInitialLR);
    this->setBatchSize(readBatchSize);
    this->setWeights(readWeights);

    std::cout << "New Model Loaded & Ready To Go!" << std::endl;
    return;

}


void MLP::printModelInfo(bool printWeights) const {

    std::cout << "Hidden Act Function: "; printMlpEnum(this->getHiddenLayerAct());
    std::cout << "Output Act Function: "; printMlpEnum(this->getOutputLayerAct());
    std::cout << "Loss Function: "; printMlpEnum(this->getLossFunc());
    std::cout << "Initial Learning Rate: " << this->initialLR << std::endl;
    std::cout << "Batch Size: " << this->batchSize << std::endl;
    std::cout << "MLP Weights Structure: " << std::endl;

    DoubleVector3D actualWeights = this->getWeights();
    std::cout << "\t Num Layers: " << actualWeights.size() << std::endl;
    for (size_t layerIdx = 0; layerIdx < actualWeights.size(); layerIdx++) {
        std::cout << "\t Matrix " << layerIdx << " Dims: (" 
            << actualWeights[layerIdx].size() << ", " << actualWeights[layerIdx][0].size() << ")" << std::endl;
    }
    if (!printWeights) {
        return;
    }
    for (size_t layerIdx = 0; layerIdx < actualWeights.size(); layerIdx++) {
        DoubleVector2D weightMatrix = actualWeights[layerIdx];
        std::cout << "Weight Matrix At Layer: " << layerIdx << std::endl;
        printMatrix(weightMatrix);
        std::cout << std::endl;
    }
    return;
}

