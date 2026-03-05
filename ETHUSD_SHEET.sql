SELECT epic, tf, ts, open, high, low, close, vol, buyers_pct, sellers_pct
FROM public.candles
where epic = 'ETHUSD'
and tf = 'M5'
ORDER BY TS DESC;
--------------------------------------------------------------------------
	
SELECT tf
FROM public.candles
WHERE epic = 'ETHUSD'
group by tf;
--------------------------------------------------------------------------	
	
select * from (
select 
epic,
tf,
to_timestamp(ts / 1000.0) as market_time,
open, 
high,
low,
close, 
vol,
buyers_pct,
sellers_pct
from candles 
)
where tf in ('M1', 'M5', 'M15', 'H1', 'H4')
and epic = 'ETHUSD'
order by market_time desc	
;
--------------------------------------------------------------------------
select * from ethusd_analytics.corr_results;
--------------------------------------------------------------------------
select * from ethusd_analytics.lagcorr_results;
--------------------------------------------------------------------------
select * from ethusd_analytics.rollingcorr_points;
--------------------------------------------------------------------------
select * from ethusd_analytics.snapshots;
--------------------------------------------------------------------------

WITH last AS (
  SELECT max(computed_at) AS t
  FROM ethusd_analytics.corr_results
  WHERE timeframe='5m'
)
SELECT *
FROM ethusd_analytics.corr_results
WHERE timeframe='5m' AND computed_at = (SELECT t FROM last)
ORDER BY abs(coalesce(pearson_r,0)) DESC
--LIMIT 100
;
--------------------------------------------------------------------------
WITH last AS (
  SELECT max(computed_at) AS t
  FROM ethusd_analytics.lagcorr_results
  WHERE timeframe='15m'
)
SELECT *
FROM ethusd_analytics.lagcorr_results
WHERE timeframe='15m' AND computed_at = (SELECT t FROM last) AND lag < 0
ORDER BY abs(coalesce(pearson_r,0)) DESC
--LIMIT 10
;
--------------------------------------------------------------------------
select * from 
ethusd_analytics.candles
where timeframe = 'tick'
order by ts desc;
--------------------------------------------------------------------------
select * from ethusd_analytics.signal_recommendations order by computed_at desc;
--------------------------------------------------------------------------
select * from ethusd_analytics.calibration_runs 
--where status = 'NO_VALID_PARAMS'
--where timeframe = '15m'
where id >= 111
order by computed_at desc;
--------------------------------------------------------------------------
select ts, epic, timeframe from ethusd_analytics.Candles group by ts, epic, timeframe;
--------------------------------------------------------------------------
select * from ethusd_analytics.Candles;
--------------------------------------------------------------------------
select * from ethusd_analytics.corr_results;
select * from ethusd_analytics.lagcorr_results;
select * from ethusd_analytics.rollingcorr_points;
select * from ethusd_analytics.snapshots;--sentiment_ticks
select * from ethusd_analytics.sentiment_ticks;--sentiment_ticks
--------------------------------------------------------------------------
-- Real OHLCV (open/high/low not null, vol not always 0)
SELECT ts, open, high, low, close, vol, sentiment_ts
FROM   ethusd_analytics.candles
WHERE  timeframe = '1m'
ORDER  BY ts DESC LIMIT 5;

-- Higher-TF resampled bars exist
SELECT timeframe, COUNT(*), MAX(ts)
FROM   ethusd_analytics.candles
GROUP  BY timeframe ORDER BY timeframe;
--------------------------------------------------------------------------
-- Sentiment ticks accumulating
SELECT ts, buyers_pct, sellers_pct
FROM   ethusd_analytics.sentiment_ticks
ORDER  BY ts DESC LIMIT 5;
--------------------------------------------------------------------------
SELECT timeframe, outcome, COUNT(*), AVG(pnl), AVG(CASE WHEN outcome='WIN' THEN 1.0 ELSE 0.0 END) win_rate
FROM ethusd_analytics.signal_recommendations
WHERE outcome IS NOT NULL
GROUP BY timeframe, outcome
ORDER BY timeframe, outcome;
--------------------------------------------------------------------------

select * from ethusd_analytics.Strategy_trades;