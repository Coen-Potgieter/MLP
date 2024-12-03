#include "mlp.h"
#include "helperfuncs.h"
#include <cmath>
#include <stdexcept>


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
/* void setWeights(const */ // TODO
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

void MLP::initWeights(InitMethod method, const int minVal, const int maxVal) {

    for (size_t layer = 0; layer < weights.size(); layer ++) {
        for (size_t row = 0; row < weights[layer].size(); row++){
            // Note the start index of col here since the first column is reserved for bias
            for (size_t col = 1; col < weights[layer][row].size(); col++){
                const double randomWeight = (rand() % ((maxVal - minVal) * 1000)) / 1000.0 + minVal;
                this->weights[layer][row][col] = randomWeight;
            }
        }
    }
}

void MLP::initBias(InitMethod method, const int minVal, const int maxVal) {
    for (size_t layer = 0; layer < this->weights.size(); layer ++) {
        for (size_t row = 0; row < this->weights[layer].size(); row++){
            // Uniform
            const double randomWeight = (rand() % ((maxVal - minVal) * 1000)) / 1000.0 + minVal;
            // Only populate the first column
            this->weights[layer][row][0] = randomWeight;
        }
    }
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

    // Apply Activation Function to each element
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

    // Transpose our Input
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

        // Save our Outputs with pre-padded row of 1s if not Output layer
        if (layer != numLayers - 1) {
            inp.insert(inp.begin(), row1s);
        }
        z.push_back(inp);

        // apply activation function and store result
        applyActivation(inp, layer);
        a.push_back(inp);
    }

    // Create an populate struct
    ForwardPropResult results;
    results.z = z;
    results.a = a;
    return results;
}

// See Obsidian Notes
void MLP::singleBackPropItter(const DoubleVector2D& inpBatch, const DoubleVector2D target) {

    // Create 3 matrcies to store nueron differentials, weight updates
    size_t numLayers = this->weights.size();
    DoubleVector3D weightUpdates(numLayers);

    DoubleVector2D layerDifferentials;
    DoubleVector2D tmpLayerDifferentials;
    
    // Forward Prop run
    ForwardPropResult resForwardProp = this->forwardProp(inpBatch);

    for (int layer = numLayers - 1; layer >= 0; layer--) {

        // Apply activation Gradient
        applyGradientActivation(resForwardProp.z[layer], layer);

        // Output Layer
        if (layer == (numLayers -1)){
            // Average Loss Gradient
            neuronDiffTmp = this->avgLossGradient(target, resForwardProp.a[layer]);
        } else {
            neuronDiffTmp = matrixMultiply(transpose(this->weights[layer+1]), neuronDiff[layer+1]);
        }
        neuronDiff[layer].push_back(elementWiseMatrixMultiply(neuronDiffTmp, resForwardProp.z[layer]));


    }
    return;


    // Output Neuron Differentials
    
    // then calc neuron differentials 
    


    // then can get weight update with those
}



 

