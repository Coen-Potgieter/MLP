#ifndef MLP_H
#define MLP_H

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <concepts>
#include <cmath>
#include <cstdint>
#include "alias.h"
#include "debug_log.h"

struct ForwardPropResult {
    DoubleVector3D z; // Output of net function
    DoubleVector3D a; // Output of activations
};

// Concept for input param as either int vector or double vector
template <typename T>
concept NumericMatrix = std::same_as<T, DoubleVector2D> || std::same_as<T, IntVector2D> || std::same_as<T, FloatVector2D>;

class MLP {

public:
    // Enum for implementation
    enum class InitMethod {
        UNIFORM,
        GAUSSIAN // TODO: Implement this
    };
    enum class ActFunc {
        SIGMOID,
        TANH,
        RELU, 
        ELU,
        SOFTMAX,
    };
    enum class LossFunc {
        SSE,
        ENTROPY // TODO: implement this
    };

    // Member Methods
    MLP(const std::vector<int>& inpStructure, const ActFunc inpHiddenLayerAct=ActFunc::RELU, const ActFunc inpOutputLayerAct=ActFunc::SIGMOID, const LossFunc inpLossFunc = LossFunc::SSE, const float inpLR=0.1, const int inpDecayRate=0, const int inpBatchSize=32);
    ~MLP();

    // Getters
    DoubleVector3D getWeights() const;
    ActFunc getHiddenLayerAct() const;
    ActFunc getOutputLayerAct() const;
    LossFunc getLossFunc() const;
    float getLR() const;
    int getDecayRate() const;
    int getBatchSize() const;
    // Setters
    void setWeights(const DoubleVector3D& inpWeights);
    void setHiddenLayerAct(const ActFunc newAct);
    void setOutputLayerAct(const ActFunc newAct);
    void setLossFunc(const LossFunc newLossFunc);
    void setLR(const float newLR);
    void setDecayRate(const int newDecayRate);
    void setBatchSize(const int newBatchSize);

    // Initialisation Methods
    void initWeights(InitMethod method, const double minVal=-1, const double maxVal=1);
    void initBias(InitMethod method, const double minVal=-1, const double maxVal=1);

    // Saving/Loading of Weights
    void saveModel(std::string_view filePath) const;
    void loadModel(std::string_view filePath);

    // Helper Functions
    void applyActivation(DoubleVector2D& Z, const size_t layer) const;
    void applyGradientActivation(DoubleVector2D& Z, const size_t layer) const;
    void printModelInfo(bool printWeights=false) const;

    // Main Functions
    ForwardPropResult forwardProp(const DoubleVector2D& inpQuery) const;
    void singleBackPropItter(const DoubleVector2D& inpBatch, const DoubleVector2D& target);
    void miniBatchGD(const DoubleVector2D& data, const DoubleVector2D& target, const size_t numEpochs=15);

    // Template For Loss Functions, See README or obsidian notes
    template <NumericMatrix Mat>
    double totalSSE(const Mat& groundTruth, const Mat& preds) const {
        // Initialise Output size 
        double loss = 0;
        double diff;
        for (size_t outputNeuronIdx = 0; outputNeuronIdx < groundTruth.size(); outputNeuronIdx++){
            for (size_t exampleIdx = 0; exampleIdx < groundTruth[0].size(); exampleIdx++) {
                diff = groundTruth[outputNeuronIdx][exampleIdx] - preds[outputNeuronIdx][exampleIdx];
                loss += 0.5 * diff * diff;
            }
        }
        return loss;
    }

    template <NumericMatrix Mat>
    double crossEntropy(const Mat& groundTruth, const Mat& preds) const {
        // Initialise Output size 
        double runningLoss = 0;
        for (size_t outputNeuronIdx = 0; outputNeuronIdx < groundTruth.size(); outputNeuronIdx++){
            for (size_t exampleIdx = 0; exampleIdx < groundTruth[0].size(); exampleIdx++) {
                
                runningLoss -= groundTruth[outputNeuronIdx][exampleIdx] * std::log(preds[outputNeuronIdx][exampleIdx] + 0.001);
            }
        }
        return runningLoss;
    }

    template <NumericMatrix Mat>
    double calcLoss(const Mat& groundTruth, const Mat& preds) const {

        // Ensure both matrices are of same size
        if ((groundTruth.size() != preds.size()) || (groundTruth[0].size() != preds[0].size())){

            DEBUG_LOG("\nExecuting calcLoss()");
            DEBUG_LOG("`groundTruth` Dimensions: (" 
                      << groundTruth.size() << ", " << groundTruth[0].size() << ")\n"
                      << "`preds` Dimensions: ("
                      << preds.size() << ", " << preds[0].size() << ")");

            throw std::invalid_argument("`groundTruth` Matrix is not of same size as `preds` matrix");
        }

        switch(this->getLossFunc()) {
            case MLP::LossFunc::SSE:
                return this->totalSSE(groundTruth, preds);
            case MLP::LossFunc::ENTROPY:
                return this->crossEntropy(groundTruth, preds);

            default:
                std::cerr << "Havent encorportated this Loss Function" << std::endl;
                return 0;
        }
    }

    template <NumericMatrix Mat>
    DoubleVector2D lossGradient(const Mat& groundTruth, const Mat& preds) const {

        // Ensure both matrices are of same size
        if ((groundTruth.size() != preds.size()) || (groundTruth[0].size() != preds[0].size())){

            DEBUG_LOG("\nExecuting avgLossGradient()");
            DEBUG_LOG("`groundTruth` Dimensions: (" 
                      << groundTruth.size() << ", " << groundTruth[0].size() << ")\n"
                      << "`preds` Dimensions: ("
                      << preds.size() << ", " << preds[0].size() << ")");
            throw std::invalid_argument("`groundTruth` Matrix is not of same size as `preds` matrix");
        }

        // Only supports SSE loss function
        const double numNeurons = groundTruth.size();
        const double numInstances = groundTruth[0].size();
        DoubleVector2D outputGrad(numNeurons, std::vector<double>(numInstances));

        for (size_t neuronIdx = 0; neuronIdx < numNeurons; neuronIdx++) {
            for (size_t exampleIdx = 0; exampleIdx < numInstances; exampleIdx++) {
                outputGrad[neuronIdx][exampleIdx] = preds[neuronIdx][exampleIdx] - groundTruth[neuronIdx][exampleIdx];
            }
        }
        return outputGrad;
    }

private:
    DoubleVector3D weights; // 3D explanation: Vector holding variable number of matrices, that are of variable size
    ActFunc hiddenLayerAct;
    ActFunc outputLayerAct;
    LossFunc lossFunc;
    float initialLR;
    int decayRate;
    int batchSize;
};
#endif
