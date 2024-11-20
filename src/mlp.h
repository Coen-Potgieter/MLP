#ifndef MLP_H
#define MLP_H

#include <vector>
#include <iostream>

class Mlp {

    private:
        std::vector<std::vector<std::vector<double>>> weights; // Vector holding variable number of matrices, that are of variable size
    public:
        Mlp(const std::vector<int>& inpStructure);
        
        ~Mlp();

        void printWeights() const;

        

};
#endif
