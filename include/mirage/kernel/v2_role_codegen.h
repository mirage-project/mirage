/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

#include "mirage/kernel/task_register.h"
#include "mirage/transpiler/utils.h"
#include <map>
#include <string>

namespace mirage {
namespace kernel {

void generate_v2_role_dispatch_code(
    mirage::transpiler::CodeKeeper &code,
    std::map<mirage::runtime::TaskType, std::string> const &task_type_to_name,
    mirage::runtime::TaskRegister const &task_register);

} // namespace kernel
} // namespace mirage
