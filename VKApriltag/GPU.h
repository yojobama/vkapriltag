//
// Created by john on 1/7/26.
//

#ifndef VKAPRILTAG_ACCELERATOR_H
#define VKAPRILTAG_ACCELERATOR_H

#include <vector>
#include <string>

namespace apriltag {
    class GPU {
    public:
        GPU(std::string name) : name(name) {};
        std::string GetName() const { return name; }
    private:
        std::string name;
    };

    std::vector<GPU> getAvailableAccelerators();
} // apriltag

#endif //VKAPRILTAG_ACCELERATOR_H