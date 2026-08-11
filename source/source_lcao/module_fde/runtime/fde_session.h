#ifndef FDE_SESSION_H
#define FDE_SESSION_H

class UnitCell;
struct Input_para;

namespace ModuleESolver
{
class ESolver;
}

namespace fde
{

/** Event-driven persistent worker for repeated fixed-fragment embedded SCFs. */
class FdeSession
{
  public:
    static void serve(ModuleESolver::ESolver& solver,
                      UnitCell& unit_cell,
                      const Input_para& input);
};

} // namespace fde

#endif // FDE_SESSION_H
