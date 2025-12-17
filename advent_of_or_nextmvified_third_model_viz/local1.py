import nextmv
from nextmv import local

# Instantiate the local application.
local_app = local.Application(src="advent_of_or_nextmvified_third_model_viz")

# Run with default inputs.
print("Running CVaR portfolio optimization with visualizations...")
run_result = local_app.new_run_with_result(
    input_dir_path="advent_of_or_nextmvified_third_model_viz/inputs",
    output_dir_path="advent_of_or_nextmvified_third_model_viz/outputs/default",
)
print(
    f"Completed run, "
    f"generated run ID: {run_result.id} with status {run_result.metadata.status_v2.value}"
)
nextmv.write(run_result)
