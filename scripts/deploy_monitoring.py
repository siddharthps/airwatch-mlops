# deploy_monitoring.py

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import IntervalSchedule
from prefect.blocks.notifications import EmailNotification
from datetime import timedelta

# Import your flow
from flows.model_monitoring import model_monitoring_flow

# Define the schedule (e.g., run every 24 hours)
# You can change this to CronSchedule for more complex schedules
# Example: Every day at 3 AM: CronSchedule(cron="0 3 * * *", timezone="America/Chicago")
monitoring_schedule = IntervalSchedule(interval=timedelta(hours=24), timezone="America/Chicago")

# Load the Email Notification block
# The name MUST match the name you used in Step 2
email_notifier_block = EmailNotification.load("model-monitoring-email-alert")

# Create the deployment
# 'name' is the deployment name in Prefect UI
# 'tags' are optional labels
# 'schedule' specifies when to run
# 'path' points to the directory containing your flow script
# 'entrypoint' is the path to your flow file relative to 'path', followed by the flow function name
# 'notifications' list the block and the states it should trigger on
deployment = Deployment.build_from_flow(
    flow=model_monitoring_flow,
    name="Air-Quality-Monitoring-Deployment",
    version="1.0",
    tags=["monitoring", "air-quality", "evidently"],
    schedule=monitoring_schedule,
    path="flows",  # The directory where model_monitoring.py lives
    entrypoint="model_monitoring.py:model_monitoring_flow",
    notifications=[
        dict(block=email_notifier_block, runs_on=["failed"]), # Trigger on FAILED state
        # You could also add:
        # dict(block=email_notifier_block, runs_on=["completed"]) # For success notifications
    ]
)

if __name__ == "__main__":
    deployment.apply()
    print("Deployment applied successfully!")
    print("Check the Prefect UI (http://127.0.0.1:4200) under 'Deployments' to see your new deployment.")
    print("Flow runs will start automatically based on the schedule.")