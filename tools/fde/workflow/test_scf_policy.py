import unittest

from workflow.scf_policy import select_adaptive_stage


STAGES = [
    {"name": "loose", "minimum_density_rms": 1e-3},
    {"name": "medium", "minimum_density_rms": 1e-5},
    {"name": "strict", "minimum_density_rms": 0.0},
]


class ScfPolicyTest(unittest.TestCase):
    def select(self, residual, history, auto):
        return select_adaptive_stage(
            STAGES, 4, residual, 20, 2, 19, history, auto)

    def test_promotes_when_outer_residual_stagnates(self):
        history = [{"density_rms": 1.0e-2},
                   {"density_rms": 9.5e-3},
                   {"density_rms": 9.2e-3}]
        result = self.select(9.2e-3, history, {
            "enabled": True, "stagnation_ratio": 0.9,
            "stagnation_cycles": 2,
        })
        self.assertEqual(result["stage"]["name"], "medium")
        self.assertEqual(result["decision"]["reason"],
                         "outer_residual_stagnation")

    def test_promotes_after_recovery_or_iteration_pressure(self):
        recovery = [{"density_rms": 2e-3, "inner_scf": {
            "F": {"retry_count": 1, "iterations": 10,
                  "maximum_iterations": 20}}}]
        result = self.select(2e-3, recovery, {"enabled": True})
        self.assertEqual(result["stage"]["name"], "medium")
        self.assertEqual(result["decision"]["reason"], "recovery_pressure")

        pressure = [{"density_rms": 2e-3, "inner_scf": {
            "F": {"retry_count": 0, "iterations": 19,
                  "maximum_iterations": 20}}}]
        result = self.select(2e-3, pressure, {"enabled": True})
        self.assertEqual(result["decision"]["reason"], "iteration_pressure")


if __name__ == "__main__":
    unittest.main()
