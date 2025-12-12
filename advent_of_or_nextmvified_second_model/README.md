# Advent of OR - Nextmv-ified

## Pre-requisites

1. Python `>=3.10` installed on your machine.

1. [Install the Nextmv CLI](https://docs.nextmv.io/docs/using-nextmv/setup/install).

1. Install the Nextmv Python SDK with additional dependencies.

   ```bash
   pip install 'nextmv[all]'
   ```

## Run the executable decision model

1. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

1. Run the code.

   ```bash
   $ python main.py

   --- Job _KVvSFtNNTVCe7sI0WPdp_Q.gms Start 12/09/25 11:56:25 52.1.0 4f802a74 DAX-DAC arm 64bit/macOS
   ... (output truncated) ...

   === Portfolio Summary ===
                  uni        value
   0       old exposure 4,068,465.00
   1       new exposure 4,587,922.10
   2          cost incr    84,483.27
   3          cost decr    67,466.92
   4             profit   878,727.34
   5         net profit   726,777.14
   6        risk weight 2,293,961.05
   7  risk weight limit 2,293,961.05
   ```

## Run locally with Nextmv

1. Run the `local1.py` script to execute 4 `local` runs with different scenarios.

   ```bash
   $ python local1.py

   Running scenario: instability_data
   Completed scenario: instability_data, generated run ID: local-emntvtf9 with status succeeded
   {
   "description": "Local run created at 2025-12-09T16:53:21.278370Z",
   "id": "local-emntvtf9",
   "metadata": {...},
   "name": "local run local-emntvtf9",
   "user_email": "",
   "console_url": ""
   }

   ... (output truncated) ...
   
   Running scenario: wealth_boom_data
   Completed scenario: wealth_boom_data, generated run ID: local-jllvb7vc with status succeeded
   {
   "description": "Local run created at 2025-12-09T16:53:31.252583Z",
   "id": "local-jllvb7vc",
   "metadata": {...},
   "name": "local run local-jllvb7vc",
   "user_email": "",
   "console_url": ""
   }
   ```

1. Make sure you have an active Nextmv account and your API key is exported.

   ```bash
   export NEXTMV_API_KEY=<YOUR_NEXTMV_API_KEY>
   ```

1. Create a Nextmv Cloud application. If you already have one created, skip
   this step. You can also do this from the Nextmv Console.

   ```bash
   $ nextmv app create -n gams-portfolio-rebalancing -a gams-portfolio-rebalancing
   
   {
      "id": "gams-portfolio-rebalancing",
      "name": "gams-portfolio-rebalancing",
      "description": "",
      "type": "custom",
      "default_instance": ""
   }
   ```

1. Run the `local2.py` script to sync the local runs to Nextmv Cloud.

   ```bash
   $ python local2.py

   ☁️ Starting sync of local application `.` to Nextmv Cloud application `gams-portfolio-rebalancing`.
   ℹ️  Found 4 local runs to sync from ./.nextmv/runs.
   🔄 Syncing local run `local-emntvtf9`...
   ✅ Synced local run `local-emntvtf9` as remote run `{'run_id': 'latest-TMBl8GGDg', 'synced_at': '2025-12-09T16:54:28.639152Z', 'app_id': 'gams-portfolio-rebalancing'}`.
   🔄 Syncing local run `local-jllvb7vc`...
   ✅ Synced local run `local-jllvb7vc` as remote run `{'run_id': 'latest-CUU_8GGvg', 'synced_at': '2025-12-09T16:54:33.391042Z', 'app_id': 'gams-portfolio-rebalancing'}`.
   🔄 Syncing local run `local-iyx3a1jt`...
   ✅ Synced local run `local-iyx3a1jt` as remote run `{'run_id': 'latest-aTC_8MMvg', 'synced_at': '2025-12-09T16:54:38.290533Z', 'app_id': 'gams-portfolio-rebalancing'}`.
   🔄 Syncing local run `local-cvi01onx`...
   ✅ Synced local run `local-cvi01onx` as remote run `{'run_id': 'latest-txiXUGGDg', 'synced_at': '2025-12-09T16:54:43.423364Z', 'app_id': 'gams-portfolio-rebalancing'}`.
   🚀 Process completed, synced local application `.` to Nextmv Cloud application `gams-portfolio-rebalancing`: 4/4 runs.
   ```

## Push to Nextmv

1. Push the code to Nextmv.

   ```bash
   $ nextmv app push -a gams-portfolio-rebalancing

   💽 Starting build for Nextmv application.
   🐍 Bundling Python dependencies.
   📋 Copied files listed in "app.yaml" manifest.
   📦 Packaged application (327.25 MiB, 25086 files).
   🌟 Pushing to application: "gams-portfolio-rebalancing".
   💥️ Successfully pushed to application: "gams-portfolio-rebalancing".
   {
      "app_id": "gams-portfolio-rebalancing",
      "endpoint": "api.cloud.nextmv.io",
      "instance_url": "https://api.cloud.nextmv.io/v1/applications/gams-portfolio-rebalancing/runs?instance_id=devint"
   }
   ```

1. Run `remotely` from the [Nextmv Console](https://cloud.nextmv.io) or the CLI.

   ```bash
   $ nextmv app run -a gams-portfolio-rebalancing -i inputs

   {
      "run_id": "devint-uzz4QMMDR"
   }
   ```
