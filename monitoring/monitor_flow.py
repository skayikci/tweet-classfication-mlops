# monitoring/monitor_flow.py
from prefect import flow, task
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@task
def run_monitor():
    """Run the monitoring script directly as a Python function instead of subprocess"""
    try:
        from monitoring.monitor import run_monitoring
        run_monitoring()
        return "✅ Monitoring completed successfully"
    except Exception as e:
        print(f"❌ Monitoring failed: {str(e)}")
        raise

@flow(name="monitoring-flow")
def monitoring_flow():
    """Main monitoring flow"""
    result = run_monitor()
    return result

if __name__ == "__main__":
    # Test the flow directly
    result = monitoring_flow()
    print(result)