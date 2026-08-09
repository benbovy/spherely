#include "s2_tmp_memory_budget.hpp"

#include <absl/flags/flag.h>

/*
** ``FLAGS_s2shape_index_tmp_memory_budget`` is defined by s2geometry and
** declared by its headers with ABSL_DECLARE_FLAG, which expands to a plain
** ``extern``.
**
** That is enough on ELF and Mach-O, and on Windows whenever s2 is linked
** statically. It is not enough against a *shared* s2, where the flag is a data
** export: CMake's WINDOWS_EXPORT_ALL_SYMBOLS does put data symbols in the
** export table -- s2.dll exports
** ``?FLAGS_s2shape_index_tmp_memory_budget@@3V?$Flag@_J@flags_internal@lts_*@absl@@A``
** from .data, and s2.lib carries the matching ``__imp_`` entry -- but for data
** an import library provides *only* that entry, so the importing side has to
** route the reference through it with ``__declspec(dllimport)``. A plain
** ``extern`` asks the linker for the bare symbol and does not link (LNK2001).
**
** Which form is right therefore depends on how s2 was built, and both happen:
** conda-forge ships a DLL, vcpkg builds s2geometry static on Windows. CMake
** decides and passes SPHERELY_S2_DLLIMPORT; see CMakeLists.txt for why the
** fallback here is the dllimport form.
**
** The two declarations cannot both be visible in one translation unit, so the
** dllimport one is made here, where no s2 header is included.
*/
#ifndef SPHERELY_S2_DLLIMPORT
#ifdef _WIN32
#define SPHERELY_S2_DLLIMPORT 1
#else
#define SPHERELY_S2_DLLIMPORT 0
#endif
#endif

#if SPHERELY_S2_DLLIMPORT
extern __declspec(dllimport) absl::Flag<std::int64_t> FLAGS_s2shape_index_tmp_memory_budget;
#else
#include <s2/mutable_s2shape_index.h>
#endif

namespace spherely {

std::int64_t get_s2_tmp_memory_budget() {
    return static_cast<std::int64_t>(absl::GetFlag(FLAGS_s2shape_index_tmp_memory_budget));
}

void set_s2_tmp_memory_budget(std::int64_t bytes) {
    absl::SetFlag(&FLAGS_s2shape_index_tmp_memory_budget, bytes);
}

}  // namespace spherely
