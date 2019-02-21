/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_HPX_CHUNKEDROUNDROBINEXECUTOR_HPP
#define KOKKOS_HPX_CHUNKEDROUNDROBINEXECUTOR_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/get_os_thread_count.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/deferred_call.hpp>

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx {
namespace parallel {
namespace execution {

///////////////////////////////////////////////////////////////////////////
/// A \a chunked_round_robin_executor creates groups of parallel execution
/// agents which execute in threads implicitly created by the executor. This
/// executor uses the scheduling hint to spawn threads with the first
/// grouped on the first core, the second group getting the next consecutive
/// threads, etc. For example, with if 10 tasks are spawned (num_tasks is
/// set to 10) and num_cores is set to 2 the executor will schedule the
/// tasks in the following order:
///
/// worker thread | 1 | 2
/// --------------+---+---
/// tasks         | 1 | 6
///               | 2 | 7
///               | 3 | 8
///               | 4 | 9
///               | 5 | 10
///
/// rather than the typical round robin:
///
/// worker thread | 1 | 2
/// --------------+---+---
/// tasks         | 1 | 2
///               | 3 | 4
///               | 5 | 6
///               | 7 | 8
///               | 9 | 10
///
/// This executor conforms to the concepts of a TwoWayExecutor,
/// and a BulkTwoWayExecutor
struct chunked_round_robin_executor {
  HPX_CONSTEXPR explicit chunked_round_robin_executor(
      std::size_t core_offset = 0,
      std::size_t num_cores = hpx::get_os_thread_count(),
      std::size_t num_tasks = std::size_t(-1))
      : core_offset_(core_offset), num_cores_(num_cores), num_tasks_(num_tasks),
        num_tasks_per_core_(double(num_tasks_) / num_cores_),
        num_tasks_spawned_(0) {}

  bool operator==(chunked_round_robin_executor const &rhs) const noexcept {
    return num_cores_ == rhs.num_cores_ && num_tasks_ == rhs.num_tasks_;
  }

  bool operator!=(chunked_round_robin_executor const &rhs) const noexcept {
    return !(*this == rhs);
  }

  chunked_round_robin_executor const &context() const noexcept { return *this; }

  template <typename F, typename... Ts> void post(F &&f, Ts &&... ts) {
    hpx::util::thread_description desc(
        f, "hpx::parallel::execution::chunked_round_robin_executor::post");

    auto const hint = threads::thread_schedule_hint(
        threads::thread_schedule_hint_mode_thread,
        core_offset_ + std::floor(double(num_tasks_spawned_ % num_tasks_) /
                                  num_tasks_per_core_));

    threads::register_thread_nullary(
        hpx::util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...),
        desc, threads::pending, false, threads::thread_priority_high, hint,
        threads::thread_stacksize_default);

    ++num_tasks_spawned_;
  }

private:
  std::size_t core_offset_;
  std::size_t num_cores_;
  std::size_t num_tasks_;
  double num_tasks_per_core_;
  mutable std::size_t num_tasks_spawned_;
};

} // namespace execution
} // namespace parallel
} // namespace hpx

namespace hpx {
namespace parallel {
namespace execution {

template <>
struct is_one_way_executor<parallel::execution::chunked_round_robin_executor>
    : std::true_type {};

} // namespace execution
} // namespace parallel
} // namespace hpx

#endif
