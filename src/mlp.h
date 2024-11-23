#ifndef MLP_H
#define MLP_H

#include <iostream>
#include <utility>
#include "alias.h"

struct ForwardPropResult {
    DoubleVector3D z; // Output of net function
    DoubleVector3D a; // Output of activations
};

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
            RELU, // TODO: Implement this
            ELU  // TODO: Implement this
        };
        enum class LossFunc {
            MSE,
            ENTROPY // TODO: implement this
        };

        // Member Methods
        MLP(const std::vector<int>& inpStructure, const ActFunc inpHiddenLayerAct=ActFunc::RELU, const ActFunc inpOutputLayerAct=ActFunc::SIGMOID, const LossFunc inpLossFunc = LossFunc::MSE, const float inpLR=0.1, const int inpDecayRate=0, const int inpBatchSize=32);
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
        /* void setWeights(const */ // TODO
        void setHiddenLayerAct(const ActFunc newAct);
        void setOutputLayerAct(const ActFunc newAct);
        void setLossFunc(const LossFunc newLossFunc);
        void setLR(const float newLR);
        void setDecayRate(const int newDecayRate);
        void setBatchSize(const int newBatchSize);

        void initWeights(InitMethod method, const int minVal=-1, const int maxVal=1);
        void initBias(InitMethod method, const int minVal=-1, const int maxVal=1);
        std::vector<double> calcError(const std::vector<double>& groundTruth, const std::vector<double>& results) const;
        std::vector<double> calcError(const std::vector<int>& groundTruth, const std::vector<int>& results) const; // Overload to handle classificaiton case
        ForwardPropResult forwardProp(DoubleVector2D inpQuery) const;
        void backPropIteration(const DoubleVector2D& inpBatch);

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
