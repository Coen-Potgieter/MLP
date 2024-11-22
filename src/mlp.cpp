#include "mlp.h"
#include "helperfuncs.h"
#include "debug_log.h"


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

std::vector<double> MLP::calcLoss(const std::vector<double>& groundTruth, const std::vector<double>& results) const {

    const int numInstances = groundTruth.size();
    std::vector<double> errors(numInstances, -1.0);

    double diff;

    for (int i = 0; i < numInstances; i++) {
        diff = groundTruth[i] - results[i];
        errors[i] = 0.5 * diff * diff;
    }
    return errors;
}

std::vector<double> MLP::calcLoss(const std::vector<int>& groundTruth, const std::vector<int>& results) const{
    std::cout << "NOT IMPLEMENTED: TODO: " << std::endl;
}

// Version 1, might change this to handle biases differently instead of always appending a vector of 1s (this may be slower)
ForwardPropResult MLP::forwardProp(DoubleVector2D inpQuery) const {

    DEBUG_LOG("Performing Forward Prop");
    
    // Row of 1s delcared outside of the for loop since row it is consistent for all iterations
    std::vector<double> row1s(inpQuery[0].size(), 1.0);

    // Declare varables to store initermediate values (See README why its 3d)
    DoubleVector3D z; 
    DoubleVector3D a; 

    // Perform forward prop
    for (int layer = 0; layer < this->weights.size(); layer++){

        // Pre-pad our inpQueryuts with row of 1s
        inpQuery.insert(inpQuery.begin(), row1s);

        // For Debugging
        DEBUG_LOG("`weights` Matrix * `inpQuery` Matrix: " 
                << this->weights[layer].size() << "x" << this->weights[layer][0].size()
                << " * " << inpQuery.size() << "x" << inpQuery[0].size());

        // wieght matrix multipled with our inpQuery and store result
        inpQuery = matrixMultiply(this->weights[layer], inpQuery);
        z.push_back(inpQuery);

        // apply activation function and store result
        tanh(inpQuery);
        a.push_back(inpQuery);
    }

    // Create an populate struct
    ForwardPropResult results;
    results.z = z;
    results.a = a;
    return results;
}

void MLP::backPropIteration(const DoubleVector2D& inpBatch) {

    // Forward prop run, then calc neuron differentials then can get weight update with those

    ForwardPropResult results = this->forwardProp(inpBatch);

}



 

