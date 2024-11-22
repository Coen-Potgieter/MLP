#ifndef MLP_H
#define MLP_H

#include <vector>
#include <iostream>
#include <utility>

class Mlp {

    private:
        std::vector<std::vector<std::vector<double>>> weights; // Vector holding variable number of matrices, that are of variable size
    public:

        // Enum to specify the weight initialisation method
        enum class InitMethod {
            UNIFORM,
            GUASSIAN
        };
        enum class ActFunc {
            SIGMOID,
            GUASSIAN
        };

        Mlp(const std::vector<int>& inpStructure);
        ~Mlp();

        void printWeights() const;
        void initWeights(InitMethod method, const int minVal=-1, const int maxVal=1);
        void initBias(InitMethod method, const int minVal=-1, const int maxVal=1);
        /* void actFunc(); */
    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>> forwardProp(std::vector<std::vector<double>> inpQuery) const;
        
        void backPropIteration(const std::vector<std::vector<double>>& inpBatch);

};
#endif
