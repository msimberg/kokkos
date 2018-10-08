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

#ifndef KOKKOS_HPX_HPP
#define KOKKOS_HPX_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_HPX)

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_HostSpace.hpp>
#include <cstddef>
#include <iosfwd>

#ifdef KOKKOS_ENABLE_HBWSPACE
#include <Kokkos_HBWSpace.hpp>
#endif

#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_TaskScheduler.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>
#include <impl/Kokkos_FunctorAnalysis.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <impl/Kokkos_Tags.hpp>
#include <impl/Kokkos_TaskQueue.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

#include <hpx/hpx_start.hpp>
#include <hpx/include/apply.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/parallel_reduce.hpp>
#include <hpx/include/run_as.hpp>
#include <hpx/include/runtime.hpp>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>



namespace Kokkos {

static bool kokkos_hpx_initialized = false;

// This represents the HPX runtime instance. It can be stateful and keep track
// of an instance of its own, but in this case it should probably just be a way
// to access properties of the HPX runtime through a common API (as defined by
// Kokkos). Can in principle create as many of these as we want and all can
// access the same HPX runtime (there can only be one in any case). Most methods
// are static.
class HPX {
public:
  using execution_space = HPX;
  using memory_space = HostSpace;
  using device_type = Kokkos::Device<execution_space, memory_space>;
  using array_layout = LayoutRight;
  using size_type = memory_space::size_type;
  using scratch_memory_space = ScratchMemorySpace<HPX>;

  inline HPX() noexcept {}
  static void print_configuration(std::ostream &, const bool verbose = false) {
    std::cout << "HPX backend" << std::endl;
  }

  // TODO: This is probably wrong.
  inline static bool in_parallel(HPX const & = HPX()) noexcept { return false; }
  inline static void fence(HPX const & = HPX()) noexcept {}

  inline static bool is_asynchronous(HPX const & = HPX()) noexcept {
    return true;
  }

  // TODO: Can this be omitted?
  static std::vector<HPX> partition(...) {}

  // TODO: What exactly does the instance represent?
  static HPX create_instance(...) { return HPX(); }

  // TODO: Can this be omitted?
  template <typename F>
  static void partition_master(F const &f, int requested_num_partitions = 0,
                               int requested_partition_size = 0) {}
  // TODO: This can get called before the runtime has been started. Still need
  // to return a reasonable value at that point.
  static int concurrency() { return hpx::get_num_worker_threads(); }

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
  static void initialize(int thread_count) {
    LOG("HPX::initialize");

    // TODO: Throw exception if initializing twice or from within the runtime?

    hpx::runtime *rt = hpx::get_runtime_ptr();
    if (rt != nullptr) {
      LOG("The HPX backend has already been initialized, skipping");
    } else {
      std::vector<std::string> config = {"hpx.os_threads=" +
                                         std::to_string(thread_count)};
      int argc_hpx = 1;
      char name[] = "kokkos_hpx";
      char *argv_hpx[] = {name, nullptr};
      hpx::start(nullptr, argc_hpx, argv_hpx, config);
      kokkos_hpx_initialized = true;
    }
  }

  static void initialize() {
    LOG("HPX::initialize");

    hpx::runtime *rt = hpx::get_runtime_ptr();
    if (rt != nullptr) {
      LOG("The HPX backend has already been initialized, skipping");
    } else {
      int argc_hpx = 1;
      char name[] = "kokkos_hpx";
      char *argv_hpx[] = {name, nullptr};
      hpx::start(nullptr, argc_hpx, argv_hpx);
      kokkos_hpx_initialized = true;
    }
  }

  static bool is_initialized() noexcept {
    LOG("HPX::is_initialized");
    return true;
    hpx::runtime *rt = hpx::get_runtime_ptr();
    return rt != nullptr;
  }

  static void finalize() {
    LOG("HPX::finalize");

    if (kokkos_hpx_initialized) {
      hpx::runtime *rt = hpx::get_runtime_ptr();
      if (rt == nullptr) {
        LOG("HPX::finalize: The backend has been stopped manually");
      } else {
        hpx::apply([]() { hpx::finalize(); });
        hpx::stop();
      }
    } else {
      LOG("HPX::finalize: the runtime was not started through Kokkos, "
          "skipping");
    }
  };

  inline static int thread_pool_size() noexcept {
    LOG("HPX::thread_pool_size");
    hpx::runtime *rt = hpx::get_runtime_ptr();
    if (rt == nullptr) {
      // TODO: Exit with error?
      return 0;
    } else {
      if (hpx::threads::get_self_ptr() == nullptr) {
        return concurrency();
      } else {
        return hpx::this_thread::get_pool()->get_os_thread_count();
      }
    }
  }

  static int thread_pool_rank() noexcept {
    LOG("HPX::thread_pool_rank");
    hpx::runtime *rt = hpx::get_runtime_ptr();
    if (rt == nullptr) {
      // TODO: Exit with error?
      return 0;
    } else {
      if (hpx::threads::get_self_ptr() == nullptr) {
        // TODO: Exit with error?
        return 0;
      } else {
        return hpx::this_thread::get_pool()->get_pool_index();
      }
    }
  }

  // TODO: What is depth? Hierarchical thread pools?
  inline static int thread_pool_size(int depth) {
    LOG("HPX::thread_pool_size");
    return 0;
  }
  static void sleep() {
    LOG("HPX::sleep");
    // TODO: Suspend the runtime?
  };
  static void wake() {
    LOG("HPX::wake");
    // TODO: Resume the runtime?
  };
  // TODO: How is this different from concurrency?
  static int get_current_max_threads() noexcept {
    LOG("HPX::get_current_max_threads");
    return concurrency();
  }
  // TODO: How is this different from concurrency?
  inline static int max_hardware_threads() noexcept {
    LOG("HPX::current_max_threads");
    return concurrency();
  }
  static int hardware_thread_id() noexcept {
    LOG("HPX::hardware_thread_id");
    return hpx::get_worker_thread_num();
  }
#else
  static void impl_initialize(int thread_count) {
    LOG("HPX::initialize");

    // TODO: Throw exception if initializing twice or from within the runtime?

    hpx::runtime *rt = hpx::get_runtime_ptr();
    if (rt != nullptr) {
      LOG("The HPX backend has already been initialized, skipping");
    } else {
      std::vector<std::string> config = {"hpx.os_threads=" +
                                         std::to_string(thread_count)};
      int argc_hpx = 1;
      char name[] = "kokkos_hpx";
      char *argv_hpx[] = {name, nullptr};
      hpx::start(nullptr, argc_hpx, argv_hpx, config);
      kokkos_hpx_initialized = true;
    }
  }

  static void impl_initialize() {
    LOG("HPX::initialize");

    hpx::runtime *rt = hpx::get_runtime_ptr();
    if (rt != nullptr) {
      LOG("The HPX backend has already been initialized, skipping");
    } else {
      int argc_hpx = 1;
      char name[] = "kokkos_hpx";
      char *argv_hpx[] = {name, nullptr};
      hpx::start(nullptr, argc_hpx, argv_hpx);
      kokkos_hpx_initialized = true;
    }
  }

  static bool impl_is_initialized() noexcept {
    LOG("HPX::impl_is_initialized");
    hpx::runtime *rt = hpx::get_runtime_ptr();
    return rt != nullptr;
  }

  static void impl_finalize() {
    LOG("HPX::finalize");

    if (kokkos_hpx_initialized) {
      hpx::runtime *rt = hpx::get_runtime_ptr();
      if (rt == nullptr) {
        LOG("HPX::finalize: The backend has been stopped manually");
      } else {
        hpx::apply([]() { hpx::finalize(); });
        hpx::stop();
      }
    } else {
      LOG("HPX::finalize: the runtime was not started through Kokkos, "
          "skipping");
    }
  };

  inline static int impl_thread_pool_size() noexcept {
    LOG("HPX::impl_thread_pool_size");
    hpx::runtime *rt = hpx::get_runtime_ptr();
    if (rt == nullptr) {
      // TODO: Exit with error?
      return 0;
    } else {
      if (hpx::threads::get_self_ptr() == nullptr) {
        return concurrency();
      } else {
        return hpx::this_thread::get_pool()->get_os_thread_count();
      }
    }
  }

  static int impl_thread_pool_rank() noexcept {
    LOG("HPX::impl_thread_pool_rank");
    hpx::runtime *rt = hpx::get_runtime_ptr();
    if (rt == nullptr) {
      // TODO: Exit with error?
      return 0;
    } else {
      if (hpx::threads::get_self_ptr() == nullptr) {
        // TODO: Exit with error?
        return 0;
      } else {
        return hpx::this_thread::get_pool()->get_pool_index();
      }
    }
  }
  inline static int impl_thread_pool_size(int depth) {
    LOG("HPX::impl_thread_pool_size");
    return 0;
  }
  inline static int impl_max_hardware_threads() noexcept {
    LOG("HPX::impl_max_hardware_threads");
    return concurrency();
  }
  KOKKOS_INLINE_FUNCTION static int impl_hardware_thread_id() noexcept {
    LOG("HPX::impl_hardware_thread_id");
    return hpx::get_worker_thread_num();
  }
#endif

  static constexpr const char *name() noexcept { return "HPX"; }
};
} // namespace Kokkos

// These specify the properties of the default memory space associate with the
// HPX execution space. Need not worry too much about this one right now.
namespace Kokkos {
namespace Impl {
template <>
struct MemorySpaceAccess<Kokkos::HPX::memory_space,
                         Kokkos::HPX::scratch_memory_space> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = false };
};

template <>
struct VerifyExecutionCanAccessMemorySpace<Kokkos::HPX::memory_space,
                                           Kokkos::HPX::scratch_memory_space> {
  enum { value = true };
  // TODO: What is this supposed to do? Does nothing in other backends as well.
  inline static void verify(void) {}
  inline static void verify(const void *) {}
};
} // namespace Impl
} // namespace Kokkos

// TODO: Is the HPX/HPXExec split necessary? How is it used in other backends?
// Serial backend does not have the split. Others have it.

// It's meant to hold instance specific data, such as scratch space allocations.
// Each backend is free to have or not have one. In the case of HPX there is not
// much value because we want to use a single HPX runtime, and we want it to be
// possible to call multiple Kokkos parallel functions at the same time. All
// allocations should thus be local to the particular invocation (meaning to the
// team member type).
namespace Kokkos {
namespace Impl {
class HPXExec {
public:
  friend class Kokkos::HPX;
  // enum { MAX_THREAD_COUNT = 512 };
  // TODO: What thread data? Data for each thread. Not really necessary for HPX.
  // void clear_thread_data() {}
  // TODO: Is this a resource partition? Check that it satisfies some criteria?
  // static void validate_partition(const int nthreads, int &num_partitions,
  //                                int &partition_size) {}

private:
  HPXExec(int arg_pool_size) {}

  // Don't want to keep team member data globally. Do it locally for each
  // invocation. More allocations but can overlap parallel regions.
  ~HPXExec() { /*clear_thread_data();*/
  }

public:
  // TODO: What assumptions can be made here? HPX allows arbitrary nesting.
  // Does this mean all threads can be master threads?
  static void verify_is_master(const char *const) {}

  // TODO: Thread = worker thread or lightweight thread i.e. task? Seems to be
  // worker thread. This one isn't really needed because we'll do all the
  // allocations locally.
  // void resize_thread_data(size_t pool_reduce_bytes, size_t team_reduce_bytes,
  //                         size_t team_shared_bytes, size_t
  //                         thread_local_bytes) {
  // }

  // This one isn't needed because we'll be doing the allocations locally.
  // inline HostThreadTeamData *get_thread_data() const noexcept {
  //   return m_pool[hpx::get_worker_thread_num()];
  // }

  // This one isn't needed because we'll be doing the allocations locally.
  // inline HostThreadTeamData *get_thread_data(int i) const noexcept {
  //   return m_pool[i];
  // }
};

} // namespace Impl
} // namespace Kokkos

// TODO: The use case of a unique token is not clear. Only used in very few
// places (scatter view?).
namespace Kokkos {
namespace Experimental {
template <> class UniqueToken<HPX, UniqueTokenScope::Instance> {
public:
  using execution_space = HPX;
  using size_type = int;
  UniqueToken(execution_space const & = execution_space()) noexcept {}
  // TODO: This could be the number of threads available to HPX.
  int size() const noexcept { return 0; }
  // TODO: This could be the worker thread id.
  int acquire() const noexcept { return 0; }
  void release(int) const noexcept {}
};

template <> class UniqueToken<HPX, UniqueTokenScope::Global> {
public:
  using execution_space = HPX;
  using size_type = int;
  UniqueToken(execution_space const & = execution_space()) noexcept {}
  // TODO: This could be the number of threads available to HPX.
  int size() const noexcept { return 0; }
  // TODO: This could be the worker thread id.
  int acquire() const noexcept { return 0; }
  void release(int) const noexcept {}
};
} // namespace Experimental
} // namespace Kokkos

// TODO: This is not complete yet.
namespace Kokkos {
namespace Impl {

// HPXTeamMember is the member_type that gets passed into user code when calling
// parallel for loops with thread team execution policies. This should provide
// enough information for the user code to determine in which thread, and team
// it is running, i.e. the indices (ranks).
//
// It should also provide access to scratch memory. The scratch memory should be
// allocated at the top level, before the first call to a parallel function, so
// that it is available for allocations in parallel code.
//
// It takes a team policy as an argument in the constructor.
struct HPXTeamMember {
private:
  typedef Kokkos::HPX execution_space;
  typedef Kokkos::ScratchMemorySpace<Kokkos::HPX> scratch_memory_space;

  // This is the actual shared scratch memory. It has two levels (0, 1).
  // Relevant on CUDA? KNL? Also contains thread specific scratch memory.
  // Scratch memory is separate from reduction memory.
  scratch_memory_space m_team_shared;
  // Size of the shared scratch memory.
  std::size_t m_team_shared_size;

  // This is the reduction buffer. It contains team_size * 512 bytes. NOTE: This
  // is also "misused" for other purposes. It can be used for the broadcast and
  // scan operations as well.
  char *m_reduce_buffer;
  std::size_t m_reduce_buffer_size;

  // int64_t *m_pool_reduce_buffer; // Exists for OpenMP backend but not used.
  // int64_t *m_pool_reduce_local_buffer;
  // int64_t *m_team_reduce_buffer;
  // int64_t *m_team_reduce_local_buffer;
  // int64_t *m_team_shared_scratch;
  // int64_t *m_thread_local_scratch;

  // Self-explanatory.
  int m_league_size;
  int m_league_rank;
  int m_team_size;
  int m_team_rank;

  // This is used to implement the barrier function.
  std::shared_ptr<hpx::lcos::local::barrier> m_barrier;

public:
  // Returns the team shared scratch memory. Exactly the same as team_scratch(0)
  // (and team_scratch(1) it seems).
  KOKKOS_INLINE_FUNCTION
  const scratch_memory_space &team_shmem() const {
    return m_team_shared.set_team_thread_mode(0, 1, 0);
  }

  // Returns the team shared scratch memory at the specified level. Level
  // ignored on CPU backends. Exactly the same as team_shmem.
  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space &team_scratch(const int) const {
    return m_team_shared.set_team_thread_mode(0, 1, 0);
  }

  // Scratch space specific for the specified thread.
  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space &thread_scratch(const int) const {
    return m_team_shared.set_team_thread_mode(0, team_size(), team_rank());
  }

  KOKKOS_INLINE_FUNCTION int league_rank() const { return m_league_rank; }
  KOKKOS_INLINE_FUNCTION int league_size() const { return m_league_size; }
  KOKKOS_INLINE_FUNCTION int team_rank() const { return m_team_rank; }
  KOKKOS_INLINE_FUNCTION int team_size() const { return m_team_size; }

  template <class... Properties>
  KOKKOS_INLINE_FUNCTION HPXTeamMember(
      const TeamPolicyInternal<Kokkos::HPX, Properties...> &policy,
      const int team_rank, const int league_rank, void *scratch,
      int scratch_size, char *reduce_buffer, std::size_t reduce_buffer_size,
      // int64_t *pool_reduce_local_buffer,
      // int64_t *team_reduce_buffer, int64_t *team_reduce_local_buffer,
      // int64_t *team_shared_scratch, int64_t *thread_local_scratch,
      std::shared_ptr<hpx::lcos::local::barrier> barrier)
      : m_team_shared(scratch, scratch_size, scratch, scratch_size),
        m_team_shared_size(scratch_size), m_league_size(policy.league_size()),
        m_league_rank(league_rank), m_team_size(policy.team_size()),
        m_team_rank(team_rank), m_reduce_buffer(reduce_buffer),
        m_reduce_buffer_size(reduce_buffer_size),
        // m_pool_reduce_local_buffer(pool_reduce_local_buffer),
        // m_team_reduce_buffer(pool_team_reduce_buffer),
        // m_team_reduce_local_buffer(team_reduce_local_buffer),
        // m_team_shared_scratch(team_shared_scratch),
        // m_thread_local_scratch(thread_local_scratch),
        m_barrier(barrier) {}

  // Waits for all team members to reach the barrier. TODO: This should also
  // issue a memory fence!?
  KOKKOS_INLINE_FUNCTION
  void team_barrier() const { m_barrier->wait(); }

  // TODO: Need to understand how the following team_* functions work in
  // relation to nested parallelism and how memory should be allocated to
  // accommodate for the temporary values.

  // TODO: Want to write value into something shared between all threads. Need
  // to take care of memory barriers here.

  // This is disabled on OpenMP backend if
  // KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST is not defined. What does that
  // do?
  template <class ValueType>
  KOKKOS_INLINE_FUNCTION void team_broadcast(ValueType &value,
                                             const int &thread_id) const {
    static_assert(std::is_trivially_default_constructible<ValueType>(),
                  "Only trivial constructible types can be broadcasted");

    // Here we simply get the beginning of the reduce buffer (same on all
    // threads) as the place to store the broadcast value.
    ValueType *const shared_value = (ValueType *)m_reduce_buffer;

    team_barrier();

    if (m_team_rank == thread_id) {
      *shared_value = value;
    }

    team_barrier();

    value = *shared_value;
  }

  template <class Closure, class ValueType>
  KOKKOS_INLINE_FUNCTION void team_broadcast(const Closure &f, ValueType &value,
                                             const int &thread_id) const {
    static_assert(std::is_trivially_default_constructible<ValueType>(),
                  "Only trivial constructible types can be broadcasted");

    // Here we simply get the beginning of the reduce buffer (same on all
    // threads) as the place to store the broadcast value.
    ValueType *const shared_value = (ValueType *)m_reduce_buffer;

    team_barrier();

    if (m_team_rank == thread_id) {
      f(value);
      *shared_value = value;
    }

    team_barrier();

    value = *shared_value;
  }

  // TODO
  template <class ValueType, class JoinOp>
  KOKKOS_INLINE_FUNCTION ValueType team_reduce(const ValueType &value,
                                               const JoinOp &op_in) const {
    // if (1 < m_team_size) {
    //   if (m_team_rank != 0) {
    //     // TODO: Magic 512.
    //     *((ValueType *)(m_reduce_buffer + m_team_rank * 512)) = value;
    //   }

    //   // Root does not overwrite shared memory until all threads arrive
    //   // and copy to their local buffer.
    //   team_barrier();

    //   if (m_team_rank == 0) {
    //     const Impl::Reducer<ValueType, JoinOp> reducer(join);

    //     ValueType *const dst = (ValueType *)m_reduce_buffer;
    //     *dst = value;

    //     for (int i = 1; i < m_team_size; ++i) {
    //       value_type *const src =
    //           (value_type *)(m_reduce_buffer + m_team_rank * 512);

    //       reducer.join(dst, *src);
    //     }
    //   }

    //   team_barrier();

    //   // TODO: Don't need to do this for team rank 0.
    //   value = *((value_type *)m_reduce_buffer);
    // }
    Kokkos::abort("HPXTeamMember: team_reduce\n");
  }

  // TODO
  template <class ReducerType>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<is_reducer<ReducerType>::value>::type
      team_reduce(const ReducerType &reducer) const {

    if (1 < m_team_size) {
      using value_type = typename ReducerType::value_type;

      if (0 != m_team_rank) {
        *((value_type *)(m_reduce_buffer + m_team_rank * 512)) =
            reducer.reference();
      }

      // Root does not overwrite shared memory until all threads arrive
      // and copy to their local buffer.
      team_barrier();

      if (0 == m_team_rank) {
        for (int i = 1; i < m_team_size; ++i) {
          value_type *const src =
              (value_type *)(m_reduce_buffer + m_team_rank * 512);

          reducer.join(reducer.reference(), *src);
        }

        *((value_type *)m_reduce_buffer) = reducer.reference();
      }

      team_barrier();

      if (0 != m_team_rank) {
        reducer.reference() = *((value_type *)m_reduce_buffer);
      }
    }
  }

  // TODO
  template <typename Type>
  KOKKOS_INLINE_FUNCTION Type
  team_scan(const Type &value, Type *const global_accum = nullptr) const {
    Kokkos::abort("HPXTeamMember: team_scan\n");
  }
};

// TeamPolicyInternal is the data that gets passed into a parallel function as a
// team policy. It should specify how many teams and threads there should be in
// the league, how much scratch space there should be.
//
// This object doesn't store any persistent state. It just holds parameters for
// parallel execution.
template <class... Properties>
class TeamPolicyInternal<Kokkos::HPX, Properties...>
    : public PolicyTraits<Properties...> {
  // These are self-explanatory.
  int m_league_size;
  int m_team_size;

  // TODO: What do these do?
  int m_team_alloc;
  int m_team_iter;

  // TODO: Are these the sizes for the two levels of scratch space? One for
  // team-shared and one for thread-specific scratch space?
  size_t m_team_scratch_size[2];
  size_t m_thread_scratch_size[2];

  // TODO: What is the chunk size? What is getting chunked? Normally this is
  // loop iterations. Can we use the HPX chunkers (static_chunk_size). Doesn't
  // really make sense though for a team policy...? Or does it mean that
  // chunk_size teams will execute immediately after each other without the
  // runtime having to go and get the next team index.
  int m_chunk_size;

  typedef TeamPolicyInternal execution_policy;
  typedef PolicyTraits<Properties...> traits;

public:
  TeamPolicyInternal &operator=(const TeamPolicyInternal &p){};

  // TODO: This should get number of threads on a single NUMA domain (in
  // current pool).
  template <class FunctorType>
  inline static int team_size_max(const FunctorType &) {
    return hpx::get_num_worker_threads();
  }

  template <class FunctorType>
  inline static int team_size_recommended(const FunctorType &) {
    return hpx::get_num_worker_threads();
  }

  template <class FunctorType>
  inline static int team_size_recommended(const FunctorType &, const int &) {
    return hpx::get_num_worker_threads();
  }

private:
  // This is just a helper function to initialize league and team sizes.
  inline void init(const int league_size_request, const int team_size_request) {
    m_league_size = league_size_request;
    const int max_team_size = hpx::get_num_worker_threads(); // team_size_max();
    m_team_size =
        team_size_request > max_team_size ? max_team_size : team_size_request;
  }

public:
  // These are just self-explanatory accessor functions.
  inline int team_size() const { return m_team_size; }
  inline int league_size() const { return m_league_size; }

  // TODO: Need to handle scratch space correctly. What is this supposed to
  // return? The scratch size of the given team on the given level? -1 means
  // shared scratch size? This is not part of the public API. This is just a
  // helper function.
  inline size_t scratch_size(const int &level, int team_size_ = -1) const {
    if (team_size_ < 0) {
      team_size_ = m_team_size;
    }
    return m_team_scratch_size[level] +
           team_size_ * m_thread_scratch_size[level];
  }

public:
  TeamPolicyInternal(typename traits::execution_space &,
                     int league_size_request, int team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0}, m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request);
  }

  TeamPolicyInternal(typename traits::execution_space &,
                     int league_size_request,
                     const Kokkos::AUTO_t &team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0}, m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    // TODO: Should we handle Kokkos::AUTO_t differently?
    init(league_size_request, hpx::get_num_worker_threads());
  }

  TeamPolicyInternal(int league_size_request, int team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0}, m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request);
  }

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t &team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0}, m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, hpx::get_num_worker_threads());
  }

  // TODO: Still don't know what these mean.
  inline int team_alloc() const { return m_team_alloc; }
  inline int team_iter() const { return m_team_iter; }
  inline int chunk_size() const { return m_chunk_size; }

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
  // These return a team policy so that the API becomes "fluent"(?). Can write
  // code like policy.set_chunk_size(...).set_scratch_size(...).
  inline TeamPolicyInternal
  set_chunk_size(typename traits::index_type chunk_size_) const {
    TeamPolicyInternal p = *this;
    p.m_chunk_size = chunk_size_;
    return p;
  }

  inline TeamPolicyInternal
  set_scratch_size(const int &level, const PerTeamValue &per_team) const {
    TeamPolicyInternal p = *this;
    p.m_team_scratch_size[level] = per_team.value;
    return p;
  }

  inline TeamPolicyInternal
  set_scratch_size(const int &level, const PerThreadValue &per_thread) const {
    TeamPolicyInternal p = *this;
    p.m_thread_scratch_size[level] = per_thread.value;
    return p;
  }

  inline TeamPolicyInternal
  set_scratch_size(const int &level, const PerTeamValue &per_team,
                   const PerThreadValue &per_thread) const {
    TeamPolicyInternal p = *this;
    p.m_team_scratch_size[level] = per_team.value;
    p.m_thread_scratch_size[level] = per_thread.value;
    return p;
  }
#else
  inline TeamPolicyInternal &
  set_chunk_size(typename traits::index_type chunk_size_) {
    m_chunk_size = chunk_size_;
    return *this;
  }

  inline TeamPolicyInternal &set_scratch_size(const int &level,
                                              const PerTeamValue &per_team) {
    m_team_scratch_size[level] = per_team.value;
    return *this;
  }

  inline TeamPolicyInternal &
  set_scratch_size(const int &level, const PerThreadValue &per_thread) {
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  inline TeamPolicyInternal &
  set_scratch_size(const int &level, const PerTeamValue &per_team,
                   const PerThreadValue &per_thread) {
    m_team_scratch_size[level] = per_team.value;
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }
#endif

  typedef HPXTeamMember member_type;
};
} // namespace Impl
} // namespace Kokkos
#endif /* #if defined( KOKKOS_ENABLE_HPX ) */
#endif /* #ifndef KOKKOS_HPX_HPP */
