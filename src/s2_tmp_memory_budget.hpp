#ifndef SPHERELY_S2_TMP_MEMORY_BUDGET_H_
#define SPHERELY_S2_TMP_MEMORY_BUDGET_H_

#include <cstdint>

namespace spherely {

/*
** Accessors for s2geometry's process-wide index-build temporary memory budget
** (``FLAGS_s2shape_index_tmp_memory_budget``, declared in
** s2/mutable_s2shape_index.h), in bytes.
**
** They are defined out of line because reaching that flag needs a declaration
** that cannot coexist with s2's own in one translation unit -- see the .cpp.
*/
std::int64_t get_s2_tmp_memory_budget();
void set_s2_tmp_memory_budget(std::int64_t bytes);

}  // namespace spherely

#endif  // SPHERELY_S2_TMP_MEMORY_BUDGET_H_
