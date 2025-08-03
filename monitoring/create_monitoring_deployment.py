# monitoring/create_monitoring_deployment.py
from prefect import serve
from monitoring.monitor_flow import monitoring_flow

if __name__ == "__main__":
    print("🚀 Starting monitoring service...")
    print("📅 Schedule: Daily at 8:00 AM Berlin time")
    print("🔄 Press Ctrl+C to stop")
    
    serve(
        monitoring_flow.to_deployment(
            name="monitoring-flow/monitoring-job",
            schedule={
                "cron": "0 8 * * *",
                "timezone": "Europe/Berlin"
            },
            tags=["monitoring"]
        ),
        limit=1
    )