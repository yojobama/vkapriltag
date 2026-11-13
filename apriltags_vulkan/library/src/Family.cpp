#include "vkapriltag/Family.h"

#include "families/FamilyTables.h"

namespace apriltag_vulkan {

namespace {

const Family *const kAll[] = {&kTag36h11,       &kTag25h9,          &kTag16h5,          &kTagCircle21h7,
                              &kTagCircle49h12, &kTagStandard41h12, &kTagStandard52h13, &kTagCustom48h12};

}  // namespace

const Family *Family::ByName(std::string_view name) {
  for (const Family *f : kAll) {
    if (name == f->name) return f;
  }
  return nullptr;
}

std::span<const Family *const> Family::All() { return kAll; }

}  // namespace apriltag_vulkan
