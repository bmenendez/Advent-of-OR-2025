import nextmv
from nextmv import local

# Instantiate the local application.
local_app = local.Application(src=".")

# Run with multiple scenarios.
scenarios = [
    "instability_data",
    "large_scale_data",
    "stress_test_data",
    "wealth_boom_data",
]
for scenario in scenarios:
    print(f"Running scenario: {scenario}")
    run_result = local_app.new_run_with_result(
        input_dir_path=f"data/{scenario}",
        output_dir_path=f"outputs/{scenario}",
    )
    print(
        f"Completed scenario: {scenario}, "
        f"generated run ID: {run_result.id} with status {run_result.metadata.status_v2.value}"
    )
    nextmv.write(run_result)
    print("\n")
