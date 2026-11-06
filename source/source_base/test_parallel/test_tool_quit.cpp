#include "source_base/parallel_global.h"
#include "source_base/parallel_reduce.h"
#include "source_base/tool_quit.h"

#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        return 2;
    }
    const std::string mode = argv[1];
    if (mode != "guards" && mode != "warning_root" && mode != "warning_last"
        && mode != "check_root" && mode != "quit_root")
    {
        return 2;
    }
    if (mode == "guards")
    {
        Parallel_Global::abort_mpi_on_error(1);
    }
    int processes = 1;
    int threads = 1;
    int rank = 0;
    Parallel_Global::read_pal_param(argc, argv, processes, threads, rank);
    if (mode == "guards")
    {
        int band_processes = 1;
        int band_rank = 0;
        int band_group = 0;
        int pool_processes = 1;
        int pool_rank = 0;
        int pool = 0;
        Parallel_Global::init_pools(processes, rank, 1, 1,
                                   band_processes, band_rank, band_group,
                                   pool_processes, pool_rank, pool);
        int diagonal_rank = 0;
        int diagonal_size = 1;
        int diagonal_color = 0;
        int grid_rank = 0;
        int grid_size = 1;
        Parallel_Global::split_diag_world(1, processes, rank,
                                         diagonal_rank, diagonal_size, diagonal_color);
        Parallel_Global::split_grid_world(1, processes, rank, grid_rank, grid_size);
        Parallel_Global::abort_mpi_on_error(0);
        Parallel_Global::finalize_mpi();
        Parallel_Global::abort_mpi_on_error(1);
        std::cerr << "ABACUS_MPI_GUARDS_OK\n";
        return 0;
    }
    int failing_rank = 0;
    if (mode == "warning_last")
    {
        failing_rank = processes - 1;
    }
    if (rank == failing_rank)
    {
        const std::string rank_text = std::to_string(rank);
        const std::string diagnostic = "ABACUS_FATAL_PROBE rank=" + rank_text + " mode=" + mode;
        std::cerr << diagnostic << std::endl;
        if (mode == "quit_root")
        {
            ModuleBase::QUIT(1);
        }
        if (mode == "check_root")
        {
            ModuleBase::CHECK_WARNING_QUIT(true, "fatal probe", "rank-local checked failure");
            return 3;
        }
        ModuleBase::WARNING_QUIT("fatal probe", "rank-local failure");
    }
    int waiting = 1;
    Parallel_Reduce::reduce_all(waiting);
    return 3;
}
