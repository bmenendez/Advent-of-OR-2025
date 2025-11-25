# Advent of OR - Nextmv-ified

## Run the executable decision model

To run this code you need Python `>=3.10`.

1. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

1. Run the code.

    ```bash
    python main.py
    ```

## Push to Nextmv

1. [Install the Nextmv CLI](https://docs.nextmv.io/docs/using-nextmv/setup/install).
2. Create a new Nextmv Application, if you already have one created, skip this
   step.

   ```bash
   nextmv app create -n <APP_NAME> -a <APP_NAME>
   ```

3. Push the code to Nextmv.

   ```bash
   nextmv app push -a <APP_NAME>
   ```

4. Run from the [Nextmv Console](https://cloud.nextmv.io) or the CLI.

   ```bash
   nextmv app run -a <APP_NAME> -i inputs
   ```
