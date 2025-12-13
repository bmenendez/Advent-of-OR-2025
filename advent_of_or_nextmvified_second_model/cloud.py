from datetime import datetime
import os

from nextmv import cloud

# Instantiate the cloud application.
client = cloud.Client(api_key=os.getenv("NEXTMV_API_KEY"))
cloud_app = cloud.Application(client=client, id="gams-portfolio-rebalancing")

version = datetime.now().strftime('%Y%m%d-%H%M%S')
cloud_app.new_version(
    name=version
)

cloud_app.push(verbose=True, app_dir=".")