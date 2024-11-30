#ifndef MLP_H
#define MLP_H

#include <iostream>
#include <stdexcept>
#include <utility>
#include <concepts>
#include "alias.h"

struct ForwardPropResult {
    DoubleVector3D z; // Output of net function
    DoubleVector3D a; // Output of activations
};

// Concept for input param as either int vector or double vector
template <typename T>
concept NumericMatrix = std::same_as<T, DoubleVector2D> || std::same_as<T, DoubleVector2D> || std::same_as<T, FloatVector2D>;

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
        void singleBackPropItter(const DoubleVector2D& inpBatch, const DoubleVector2D target);
        
        // Template
        template <NumericMatrix Mat>
        std::vector<double> calcLoss(const Mat& groundTruth, const Mat& preds) const {

            // SSE loss function
            const int numInstances = groundTruth[0].size();
            const int numNeurons = groundTruth.size();
            std::vector<double> neuronLosses(numNeurons);
            double diff;
            double runningLoss;
            for (size_t neuronIdx = 0; neuronIdx < numNeurons; neuronIdx++){
                runningLoss = 0;
                for (size_t exampleIdx = 0; exampleIdx < numInstances; exampleIdx++) {
                    diff = groundTruth[neuronIdx][exampleIdx] - preds[neuronIdx][exampleIdx];
                    runningLoss += 0.5 * diff * diff;
                }
                neuronLosses[neuronIdx] = runningLoss / numInstances;
            }
            return neuronLosses;
        }

        template <NumericMatrix Mat>
        DoubleVector2D avgLoss(const Mat& groundTruth, const Mat& preds) const {

            // Ensure both matrices are of same size
            if ((groundTruth.size() != preds.size()) || (groundTruth[0].size() != preds[0].size())){
                throw std::invalid_argument("`groundTruth` Matrix is not of same size as `preds` matrix");
            }

            // Only supports MLE loss function
            const double numInstances = groundTruth[0].size();
            DoubleVector2D outp(groundTruth.size(), groundTruth[0].size());
            std::cout << outp.size() << std::endl;
            std::cout << outp[0].size() << std::endl;

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
