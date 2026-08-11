#include "fde_session.h"
#include "fde_session_protocol.h"
#include "source_lcao/module_fde/runtime/fde_warm_start.h"

#include "source_esolver/esolver.h"
#include "source_esolver/esolver_ks_lcao.h"
#include "source_io/module_parameter/input_parameter.h"

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef __MPI
#include <mpi.h>
#endif

namespace fde
{

namespace
{

std::string sanitize_message(const std::string& message)
{
    std::string result(message);
    std::replace(result.begin(), result.end(), '\n', ' ');
    std::replace(result.begin(), result.end(), '\r', ' ');
    return result;
}

#ifdef __MPI
void require_mpi(const int result, const char* operation)
{
    if (result != MPI_SUCCESS)
    {
        throw std::runtime_error(std::string("FDE session MPI failure while ")
                                 + operation);
    }
}

void broadcast_line(std::string& line, const int rank)
{
    unsigned long long size
        = rank == 0 ? static_cast<unsigned long long>(line.size()) : 0ULL;
    require_mpi(MPI_Bcast(&size,
                          1,
                          MPI_UNSIGNED_LONG_LONG,
                          0,
                          MPI_COMM_WORLD),
                "broadcasting command size");
    if (size > static_cast<unsigned long long>(
                   std::numeric_limits<int>::max()))
    {
        throw std::length_error("FDE session command exceeds MPI count range");
    }
    if (rank != 0)
    {
        line.resize(static_cast<std::size_t>(size));
    }
    if (size > 0)
    {
        require_mpi(MPI_Bcast(&line[0],
                              static_cast<int>(size),
                              MPI_CHAR,
                              0,
                              MPI_COMM_WORLD),
                    "broadcasting command text");
    }
}
#endif

template <typename TK, typename TR>
bool run_request(ModuleESolver::ESolver& solver,
                 UnitCell& unit_cell,
                 const FdeSessionCommand& command,
                 const FdeWarmStartPlan& warm_start)
{
    ModuleESolver::ESolver_KS_LCAO<TK, TR>* lcao
        = dynamic_cast<ModuleESolver::ESolver_KS_LCAO<TK, TR>*>(&solver);
    if (lcao == nullptr)
    {
        return false;
    }
    lcao->reload_fde_session(command.config_path, unit_cell);
    lcao->runner(unit_cell, warm_start.ionic_step);
    return true;
}

} // namespace

void FdeSession::serve(ModuleESolver::ESolver& solver,
                       UnitCell& unit_cell,
                       const Input_para& input)
{
    if (input.fde_task != "embedded_session")
    {
        throw std::invalid_argument(
            "FdeSession requires fde_task embedded_session");
    }
    int rank = 0;
#ifdef __MPI
    require_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "querying world rank");
#endif
    if (rank == 0)
    {
        std::cout << "FDE_SESSION_READY 1" << std::endl;
    }

    std::size_t request_index = 0;
    while (true)
    {
        std::string line;
        if (rank == 0 && !std::getline(std::cin, line))
        {
            line = "STOP";
        }
#ifdef __MPI
        broadcast_line(line, rank);
#endif
        FdeSessionCommand command;
        try
        {
            command = FdeSessionProtocol::parse(line);
        }
        catch (const std::exception& error)
        {
            if (rank == 0)
            {
                std::cout << "FDE_SESSION_ERROR protocol "
                          << sanitize_message(error.what()) << std::endl;
            }
            return;
        }
        if (command.type == FdeSessionCommand::stop)
        {
            if (rank == 0)
            {
                std::cout << "FDE_SESSION_STOPPED" << std::endl;
            }
            return;
        }

        std::string local_error;
        bool handled = false;
        FdeWarmStartPlan warm_start;
        try
        {
            warm_start = FdeWarmStart::plan(request_index);
            handled = run_request<double, double>(solver,
                                                   unit_cell,
                                                   command,
                                                   warm_start)
                      || run_request<std::complex<double>, double>(
                          solver, unit_cell, command, warm_start)
                      || run_request<std::complex<double>,
                                     std::complex<double> >(
                          solver, unit_cell, command, warm_start);
            if (!handled)
            {
                throw std::runtime_error(
                    "FDE session supports only LCAO KS solvers");
            }
        }
        catch (const std::exception& error)
        {
            local_error = error.what();
        }
        int success = local_error.empty() ? 1 : 0;
#ifdef __MPI
        int global_success = 0;
        require_mpi(MPI_Allreduce(&success,
                                  &global_success,
                                  1,
                                  MPI_INT,
                                  MPI_MIN,
                                  MPI_COMM_WORLD),
                    "reducing request status");
        success = global_success;
#endif
        if (success == 0)
        {
            if (rank == 0)
            {
                const std::string message
                    = local_error.empty() ? "request failed on another MPI rank"
                                          : local_error;
                std::cout << "FDE_SESSION_ERROR " << command.request_id << ' '
                          << sanitize_message(message) << std::endl;
            }
            return;
        }
#ifdef __MPI
        require_mpi(MPI_Barrier(MPI_COMM_WORLD), "completing request");
#endif
        if (rank == 0)
        {
            std::cout << "FDE_SESSION_DONE " << command.request_id << ' '
                      << warm_start.mode << ' ' << warm_start.ionic_step
                      << std::endl;
        }
        ++request_index;
    }
}

} // namespace fde
