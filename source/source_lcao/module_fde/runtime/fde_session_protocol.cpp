#include "fde_session_protocol.h"

#include <sstream>
#include <stdexcept>
#include <string>

namespace fde
{

namespace
{

std::string protocol_error(const std::string& message)
{
    return "invalid FDE session command: " + message;
}

} // namespace

FdeSessionCommand FdeSessionProtocol::parse(const std::string& line)
{
    std::istringstream input(line);
    std::string operation;
    input >> operation;
    if (operation == "STOP")
    {
        std::string extra;
        if (input >> extra)
        {
            throw std::invalid_argument(protocol_error("STOP takes no arguments"));
        }
        FdeSessionCommand command;
        command.type = FdeSessionCommand::stop;
        return command;
    }
    if (operation != "RUN")
    {
        throw std::invalid_argument(protocol_error("expected RUN or STOP"));
    }
    FdeSessionCommand command;
    command.type = FdeSessionCommand::run;
    if (!(input >> command.request_id >> command.config_path))
    {
        throw std::invalid_argument(
            protocol_error("RUN requires request_id and config_path"));
    }
    std::string extra;
    if (input >> extra)
    {
        throw std::invalid_argument(protocol_error("RUN has extra fields"));
    }
    if (command.request_id.empty() || command.config_path.empty()
        || command.config_path[0] != '/')
    {
        throw std::invalid_argument(
            protocol_error("request_id must be nonempty and config_path absolute"));
    }
    return command;
}

} // namespace fde
