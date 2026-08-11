import unittest

from workflow.profiler import aggregate_profiles, subsystem_phase_profile


class ProfilerTest(unittest.TestCase):
    def test_separates_electronic_and_abacus_overhead(self):
        profile = subsystem_phase_profile({
            "workflow_preparation_seconds": 2.0,
            "session_startup_seconds": 3.0,
            "wall_time_seconds": 10.0,
            "electronic_step_time_seconds": 6.0,
            "artifact_validation_seconds": 1.0,
        })
        self.assertEqual(profile["electronic_steps"], 6.0)
        self.assertEqual(profile["abacus_overhead"], 4.0)
        self.assertEqual(sum(profile.values()), 16.0)

    def test_aggregates_reuse_and_retry_diagnostics(self):
        summary = aggregate_profiles([
            {"wall_time_seconds": 5.0, "electronic_step_time_seconds": 4.0,
             "session_reused": False, "retry_count": 0},
            {"wall_time_seconds": 4.0, "electronic_step_time_seconds": 3.0,
             "session_reused": True, "retry_count": 1},
        ])
        self.assertEqual(summary["total_retries"], 1)
        self.assertEqual(summary["session_reuse_fraction"], 0.5)
        self.assertAlmostEqual(
            summary["electronic_step_fraction_of_abacus"], 7.0 / 9.0)


if __name__ == "__main__":
    unittest.main()
