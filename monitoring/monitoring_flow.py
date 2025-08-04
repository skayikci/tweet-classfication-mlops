# e/monitoring/monitor_flow.py

from prefect import flow, task
import subprocess
import os
import logging

log_path = os.path.join(os.path.dirname(__file__), "logs", "monitor.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.INFO)


@task
def run_generate_sets():
    logging.info("🔧 Generating monitoring sets...")
    subprocess.run(["python", "monitoring/generate_monitoring_sets.py"], check=True)
    logging.info("✅ Monitoring sets generated.")


@task
def run_drift_report():
    logging.info("📊 Generating drift report...")
    subprocess.run(["python", "monitoring/generate_drift_report.py"], check=True)
    logging.info("✅ Drift report generated.")


@task
def find_latest_report() -> str:
    reports_dir = os.path.join("monitoring", "reports")
    reports = [
        f
        for f in os.listdir(reports_dir)
        if f.startswith("drift_report_") and f.endswith(".html")
    ]
    reports.sort()
    if reports:
        latest = reports[-1]
        logging.info(f"📝 Latest drift report: {latest}")
        return os.path.join(reports_dir, latest)
    else:
        raise FileNotFoundError("No drift reports found.")


@flow(name="monitoring-flow")
def monitoring_flow():
    logging.info("🚀 Monitoring flow started.")
    run_generate_sets()
    run_drift_report()
    latest_report = find_latest_report()
    logging.info(f"📎 Monitoring flow finished. Latest report: {latest_report}")


if __name__ == "__main__":
    monitoring_flow()
