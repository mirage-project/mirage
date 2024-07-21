#include "mirage/layout.h"

namespace mirage {
namespace layout {

CmemLayout dmemlayout_to_cmemlayout(DmemLayout dmem_layout) {
  switch (dmem_layout) {
    case DmemLayout::DmemRowMajor:
      return CmemLayout::CmemRowMajor;
    case DmemLayout::DmemColumnMajor:
      return CmemLayout::CmemColumnMajor;
    default:
      return CmemLayout::CmemUnknownLayout;
  }
}

} // namespace layout
} // namespace mirage
