#include "../runtime/fde_session_protocol.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace
{

TEST(FdeSessionProtocol, ParsesRunAndStop)
{
    const fde::FdeSessionCommand run
        = fde::FdeSessionProtocol::parse("RUN cycle-001 /tmp/FDE_CONFIG");
    EXPECT_EQ(run.type, fde::FdeSessionCommand::run);
    EXPECT_EQ(run.request_id, "cycle-001");
    EXPECT_EQ(run.config_path, "/tmp/FDE_CONFIG");

    const fde::FdeSessionCommand stop
        = fde::FdeSessionProtocol::parse("STOP");
    EXPECT_EQ(stop.type, fde::FdeSessionCommand::stop);
}

TEST(FdeSessionProtocol, RejectsRelativePathsAndExtraFields)
{
    EXPECT_THROW(
        fde::FdeSessionProtocol::parse("RUN cycle-001 FDE_CONFIG"),
        std::invalid_argument);
    EXPECT_THROW(
        fde::FdeSessionProtocol::parse("STOP now"),
        std::invalid_argument);
}

} // namespace
