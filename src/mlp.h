#ifndef MLP_H
#define MLP_H

#include <iostream>
#include <utility>
#include <concepts>
#include "alias.h"

struct ForwardPropResult {
    DoubleVector3D z; // Output of net function
    DoubleVector3D a; // Output of activations
};

// Concept for input param as either int vector or double vector
template <typename T>
concept NumericVector = std::same_as<T, std::vector<double>> || std::same_as<T, std::vector<int>> || std::same_as<T, std::vector<float>>;

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
        ForwardPropResult forwardProp(const DoubleVector2D& inpQuery) const;
        void backPropIteration(const DoubleVector2D& inpBatch);
        
        // Template
        template <NumericVector Vec>
        std::vector<double> calcError(const Vec& groundTruth, const Vec& preds) const {
            const int numInstances = groundTruth.size();
            std::vector<double> errors(numInstances, -1.0);

            double diff;

            for (int i = 0; i < numInstances; i++) {
                diff = groundTruth[i] - preds[i];
                errors[i] = 0.5 * diff * diff;
            }
            return errors;
        }
        template <NumericVector Vec>
        double calcAvgLoss(const Vec& groundTruth, const Vec& preds) const {

            const double numInstances = groundTruth.size();
            double runningSum = 0;
            for (int i = 0; i < numInstances; i++) {
                runningSum += preds[i] - groundTruth[i];
            }
            return runningSum / numInstances;
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
