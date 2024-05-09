#include "mirage/profile_result.h"

#include <limits>

namespace mirage {

ProfileResult ProfileResult::infinity() {
  return ProfileResult{1000};
}

}
