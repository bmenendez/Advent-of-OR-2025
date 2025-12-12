import os

from nextmv import cloud, local

# Instantiate the cloud application.
client = cloud.Client(api_key=os.getenv("NEXTMV_API_KEY"))
cloud_app = cloud.Application(client=client, id="gams-portfolio-rebalancing-covariance")

# Instantiate the local application.
local_app = local.Application(src="advent_of_or_nextmvified_second_model")

# Sync the local application to the cloud.
local_app.sync(target=cloud_app, verbose=True)
