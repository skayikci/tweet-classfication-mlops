CREATE TABLE prediction_logs (
  id SERIAL PRIMARY KEY,
  input_text TEXT NOT NULL,
  predicted_label TEXT NOT NULL,
  user_label TEXT,
  confidence FLOAT,
  timestamp TIMESTAMP DEFAULT NOW(),
  CONSTRAINT unique_input_text_predicted_label UNIQUE (input_text, predicted_label)
);


CREATE INDEX idx_prediction_logs_timestamp ON prediction_logs (timestamp);
CREATE INDEX idx_prediction_logs_label ON prediction_logs (predicted_label);
CREATE INDEX idx_prediction_logs_user_label ON prediction_logs (user_label);
CREATE OR REPLACE FUNCTION log_prediction(
    p_input_text TEXT,
    p_predicted_label TEXT,
    p_user_label TEXT DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
    INSERT INTO prediction_logs (input_text, predicted_label, user_label)
    VALUES (p_input_text, p_predicted_label, p_user_label);
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION get_predictions_by_label(
    p_label TEXT
) RETURNS TABLE (
    id INT,
    input_text TEXT,
    predicted_label TEXT,
    user_label TEXT,
    timestamp TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT id, input_text, predicted_label, user_label, timestamp
    FROM prediction_logs
    WHERE predicted_label = p_label
    ORDER BY timestamp DESC;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION get_predictions_by_user_label(
    p_user_label TEXT
) RETURNS TABLE (
    id INT,
    input_text TEXT,
    predicted_label TEXT,
    user_label TEXT,
    timestamp TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT id, input_text, predicted_label, user_label, timestamp
    FROM prediction_logs
    WHERE user_label = p_user_label
    ORDER BY timestamp DESC;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION get_predictions_by_text(
    p_input_text TEXT
) RETURNS TABLE (
    id INT,
    input_text TEXT,
    predicted_label TEXT,
    user_label TEXT,
    timestamp TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT id, input_text, predicted_label, user_label, timestamp
    FROM prediction_logs
    WHERE input_text ILIKE '%' || p_input_text || '%'
    ORDER BY timestamp DESC;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION get_predictions_by_date_range(
    p_start_date TIMESTAMP,
    p_end_date TIMESTAMP
) RETURNS TABLE (
    id INT,
    input_text TEXT,
    predicted_label TEXT,
    user_label TEXT,
    timestamp TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT id, input_text, predicted_label, user_label, timestamp
    FROM prediction_logs
    WHERE timestamp BETWEEN p_start_date AND p_end_date
    ORDER BY timestamp DESC;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION get_all_predictions()
RETURNS TABLE (
    id INT,
    input_text TEXT,
    predicted_label TEXT,
    user_label TEXT,
    timestamp TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT id, input_text, predicted_label, user_label, timestamp
    FROM prediction_logs
    ORDER BY timestamp DESC;
END;
$$ LANGUAGE plpgsql;