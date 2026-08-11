#ifndef FDE_SESSION_PROTOCOL_H
#define FDE_SESSION_PROTOCOL_H

#include <string>

namespace fde
{

struct FdeSessionCommand
{
    enum Type
    {
        run,
        stop
    };

    Type type;
    std::string request_id;
    std::string config_path;
};

class FdeSessionProtocol
{
  public:
    static FdeSessionCommand parse(const std::string& line);
};

} // namespace fde

#endif // FDE_SESSION_PROTOCOL_H
