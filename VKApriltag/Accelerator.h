//
// Created by john on 1/7/26.
//

#ifndef VKAPRILTAG_ACCELERATOR_H
#define VKAPRILTAG_ACCELERATOR_H

#include <vector>
#include <string>

namespace apriltag {
    class Accelerator {
    public:
        std::string GetName() const { return name; }
    private:
        std::string name;
    };

    std::vector<Accelerator> getAvailableAccelerators();
} // apriltag

#endif //VKAPRILTAG_ACCELERATOR_H