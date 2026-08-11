#ifndef FDE_PES_SCAN_H
#define FDE_PES_SCAN_H

#include "source_lcao/module_fde/orchestration/fde_freeze_thaw.h"

#include <iosfwd>
#include <string>
#include <vector>

namespace fde
{

struct GeometryPoint
{
    std::string label;
    std::string geometry_fingerprint;
    double coordinate_bohr;
};

struct DiabaticStateDefinition
{
    std::string label;
    std::string first_fragment_label;
    std::string second_fragment_label;
    int first_alpha_electrons;
    int first_beta_electrons;
    int second_alpha_electrons;
    int second_beta_electrons;
};

struct DiabaticStateRunResult
{
    FreezeThawCheckpoint checkpoint;
    double localization_score;
};

class DiabaticStateRunner
{
  public:
    virtual ~DiabaticStateRunner() {}

    virtual DiabaticStateRunResult run(
        const GeometryPoint& geometry,
        const DiabaticStateDefinition& state,
        const FreezeThawCheckpoint* same_state_warm_start) const = 0;
};

struct PesScanControls
{
    double electron_tolerance;
    double minimum_localization_score;
};

struct DiabaticPesPoint
{
    std::string geometry_label;
    std::string geometry_fingerprint;
    double coordinate_bohr;
    std::string state_label;
    double energy_ry;
    double localization_score;
    int freeze_thaw_cycles;
};

struct DiabaticPesDiagnostics
{
    double minimum_localization_score;
    double minimum_absolute_gap_ry;
    double maximum_adjacent_energy_change_ry;
    double maximum_slope_change_ry_per_bohr;
    int crossing_brackets;
};

struct DiabaticPes
{
    int schema_version;
    std::string first_state_label;
    std::string second_state_label;
    std::vector<DiabaticPesPoint> points;
    DiabaticPesDiagnostics diagnostics;
};

class TwoStatePesScan
{
  public:
    static DiabaticPes run(
        const std::vector<GeometryPoint>& geometries,
        const std::vector<DiabaticStateDefinition>& states,
        const PesScanControls& controls,
        const DiabaticStateRunner& runner);

    static void write(std::ostream& output, const DiabaticPes& pes);
};

} // namespace fde

#endif // FDE_PES_SCAN_H
