-- Add Greeks and contract times to kalshi_orderbook_snapshots table

ALTER TABLE kalshi_orderbook_snapshots
ADD COLUMN IF NOT EXISTS contract_start_time TIMESTAMP,
ADD COLUMN IF NOT EXISTS contract_end_time TIMESTAMP,
ADD COLUMN IF NOT EXISTS delta DECIMAL(10, 6),
ADD COLUMN IF NOT EXISTS gamma DECIMAL(10, 6),
ADD COLUMN IF NOT EXISTS vega DECIMAL(10, 6),
ADD COLUMN IF NOT EXISTS theta DECIMAL(10, 6);

-- Add index on contract_end_time for expiry queries
CREATE INDEX IF NOT EXISTS idx_kalshi_ob_contract_end ON kalshi_orderbook_snapshots(contract_end_time);
