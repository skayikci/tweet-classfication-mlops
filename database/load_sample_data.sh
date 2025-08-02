#!/bin/bash
# Import JSONL into the prediction_logs table

echo "⏳ Importing sample data into prediction_logs..."

psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\
COPY prediction_logs(input_text, predicted_label, user_label, timestamp)
FROM PROGRAM 'jq -r \"[.input_text, .predicted_label, .user_label, .timestamp] | @csv\" /docker-entrypoint-initdb.d/sample_prediction_logs_with_user_labels.jsonl'
WITH CSV;"

echo "✅ Sample data import complete!"
