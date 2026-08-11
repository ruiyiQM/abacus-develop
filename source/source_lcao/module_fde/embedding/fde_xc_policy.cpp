#include "source_lcao/module_fde/embedding/fde_xc_policy.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace fde
{
namespace
{

std::string lowercase(const std::string& value)
{
    std::string result(value);
    std::transform(result.begin(),
                   result.end(),
                   result.begin(),
                   [](const unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    return result;
}

} // namespace

std::string FdeXcPolicy::canonical_fragment(const std::string& name)
{
    const std::string canonical = lowercase(name);
    if (canonical != "pbe" && canonical != "pbe0" && canonical != "scan")
    {
        throw std::invalid_argument(
            "FDE fragment XC must be pbe, pbe0, or scan");
    }
    return canonical;
}

std::string FdeXcPolicy::canonical_embedding(const std::string& name)
{
    const std::string canonical = lowercase(name);
    if (canonical != "pbe")
    {
        throw std::invalid_argument(
            "FDE embedding XC currently supports only pbe; fragment_xc may "
            "independently select pbe, pbe0, or scan");
    }
    return canonical;
}

} // namespace fde
