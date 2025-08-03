#!/bin/bash
echo "⏳ Importing sample data into prediction_logs..."
psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\
    COPY prediction_logs(input_text, predicted_label, user_label, timestamp)
    FROM '/docker-entrypoint-initdb.d/003_sample_prediction_logs.csv'
    WITH CSV HEADER;"
echo "✅ Sample data import complete!"
