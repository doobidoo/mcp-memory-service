-- #1506 part 2: the pull-side echo guard needs the window a forward was in flight.
-- started_at: written BEFORE the secondary is called (earliest attempt kept), so a
-- pull that runs mid-forward, or after a crash between the secondary's ack and the
-- hub's record, still recognises the D1 stamp as the hub's own echo.
-- acked: 0 while the forward is pending, 1 once the secondary acknowledged.
-- Rows written by part 1 keep started_at NULL and are not used by the guard.
ALTER TABLE secondary_forwarded ADD COLUMN started_at REAL;
ALTER TABLE secondary_forwarded ADD COLUMN acked INTEGER NOT NULL DEFAULT 1;
