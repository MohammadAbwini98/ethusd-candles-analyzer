# Trading Chart UI (React + Lightweight Charts)

A high-performance, **Capital.com-like** trading chart component built with React, TypeScript, TailwindCSS, and Lightweight Charts.

## Features

✅ **Live updates without jank** – bars update with `series.update()`, not `setData()`  
✅ **Range selectors** – 1D, 7D, 30D, 3M, 6M, 1Y with auto resolution mapping  
✅ **Lazy left-load** – scroll past left edge to fetch older bars (prepend)  
✅ **State preservation** – zoom, pan, visible range persisted to localStorage  
✅ **Chart types** – Candlestick & Line  
✅ **Indicators** – EMA(20/50/200), ATR(14) with instant toggle  
✅ **Dark/Light themes** – instant toggle, persisted  
✅ **Volume subplot** – toggle on/off  
✅ **Responsive** – ResizeObserver, mobile-friendly controls  
✅ **Crosshair OHLC readout** – real-time hover updates  

---

## Quick Start

### 1. Install Dependencies

```bash
cd trading-chart-ui
npm install
```

### 2. Run Development Server

```bash
npm run dev
```

Open `http://localhost:5173/` in your browser.

### 3. Build for Production

```bash
npm run build
```

Output goes to `dist/`.

---

## Architecture

### File Structure

```
src/
├── main.tsx                 # App entry point
├── App.tsx                  # Root component
├── styles.css              # Global Tailwind styles
├── types.ts                # Core TypeScript interfaces
├── store/
│   └── chartStore.ts       # Zustand persisted state (preferences)
├── components/
│   ├── TradingChart.tsx    # Orchestrator (connects all controllers + UI)
│   ├── ChartContainer.tsx  # Imperative chart wrapper (forwardRef)
│   └── UIControls.tsx      # Range/type/theme/indicator controls
├── controllers/
│   ├── DataController.ts   # Range loading, cache merge, lazy prepend
│   └── RealtimeController.ts # Live tick subscription (rAF-throttled)
├── data/
│   ├── barCache.ts         # Memory + IndexedDB cache
│   └── mockAdapters.ts     # Mock REST/WebSocket adapters
└── utils/
    ├── time.ts             # Range→resolution, bar bucketing, merge/trim
    └── indicators.ts       # EMA, ATR calculations
```

### State Flow

```
TradingChart (orchestrator)
  ├─→ useChartStore (persisted: range, theme, indicators, etc.)
  ├─→ ChartContainer (ref-based imperative chart handle)
  ├─→ UIControls (triggers range/theme/indicator changes)
  ├─→ DataController (fetch → cache → merge → prepend)
  └─→ RealtimeController (subscribe → throttle → update last bar)
```

### Key Decisions

1. **Imperative Chart Ref** – Prevents unnecessary Lightweight Charts re-initialization. State updates only trigger chart method calls, not remounting.
2. **rAF-Throttled Ticks** – Live updates queued and flushed at ~20fps max to avoid 60fps+ rendering.
3. **Lazy Left-Load** – When user scrolls to left edge, background task fetches older bars and prepends without resetting viewport.
4. **Auto Resolution Mapping** – Ranges automatically select resolutions (1D→1m, 7D→5m, 1Y→1d) to keep bar count reasonable.
5. **Persisted Preferences** – Theme, indicators, last visible range stored in localStorage via Zustand.
6. **Unified Adapter Interfaces** – `DataAdapter` and `RealtimeAdapter` are mocked; easy to plug in Capital.com APIs.

---

## Plugging in Real APIs

### 1. Replace `mockDataAdapter` (Historical Bars)

Edit `src/data/mockAdapters.ts`:

```typescript
export const realDataAdapter: DataAdapter = {
  async fetchBars(params) {
    const response = await fetch(
      `/api/candles?symbol=${params.symbol}&resolution=${params.resolution}&from=${params.from}&to=${params.to}&limit=${params.limit}`
    );
    const data = await response.json();
    return data.items; // Expect: [{ time, open, high, low, close, volume }, ...]
  },
};
```

Then update `src/components/TradingChart.tsx`:

```typescript
import { realDataAdapter } from "../data/mockAdapters"; // ← change here
const dataControllerRef = useRef(new DataController(realDataAdapter));
```

### 2. Replace `mockRealtimeAdapter` (Live Ticks)

```typescript
export const realRealtimeAdapter: RealtimeAdapter = {
  subscribeTicks(symbol, onTick) {
    const ws = new WebSocket(`wss://api.capital.com/ws?symbol=${symbol}`);
    ws.onmessage = (event) => {
      const { time, bid, ask, last } = JSON.parse(event.data);
      onTick({ time: time * 1000, bid, ask, last }); // time expected in ms
    };
    return {
      unsubscribe: () => ws.close(),
    };
  },
};
```

Update `src/components/TradingChart.tsx`:

```typescript
import { realRealtimeAdapter } from "../data/mockAdapters"; // ← change here
const realtimeControllerRef = useRef(new RealtimeController(realRealtimeAdapter));
```

### 3. Payload Format Expectations

**Historical bars** (from `fetchBars()`):

```json
{
  "items": [
    {
      "time": 1640995200,
      "open": 1800.5,
      "high": 1805.0,
      "low": 1799.2,
      "close": 1803.8,
      "volume": 5400
    }
  ]
}
```

**Live ticks** (via `onTick()`):

```json
{
  "time": 1641081600000,
  "bid": 1803.75,
  "ask": 1803.85,
  "last": 1803.80
}
```

---

## Usage Examples

### Change Default Symbol

Edit `src/App.tsx`:

```typescript
export function App() {
  return <TradingChart symbol="BTCUSD" />;
}
```

### Add Custom Indicator

Edit `src/utils/indicators.ts`:

```typescript
export function rsiSeries(bars: Bar[], period = 14): IndicatorPoint[] {
  // ... your RSI calculation
}
```

Add to `ChartContainer.tsx`:

```typescript
const rsiRef = useRef<any>(null);
rsiRef.current = chart.addLineSeries({ color: "#9333ea", lineWidth: 1 });
```

Update `types.ts`:

```typescript
export interface IndicatorsState {
  ema20: boolean;
  ema50: boolean;
  ema200: boolean;
  atr14: boolean;
  rsi14: boolean; // ← add
}
```

---

## Performance Tips

1. **Limit bars in memory** – Cache trimmed to 10,000 bars per {symbol, resolution}
2. **Throttle live updates** – rAF-queued, max 20 updates/sec
3. **Lazy load threshold** – triggers at 30 bars from left edge
4. **ResizeObserver** – auto-resizes chart on window/container resize
5. **Theme colors memoized** – recalculated only when theme changes

---

## Testing / Acceptance Criteria

Run dev server and verify:

1. ✅ **Zoom in, pan left** → Chart stays. Scroll far left → loads older bars.
2. ✅ **Switch 1D → 7D** → Smooth transition. Settings preserved.
3. ✅ **Enable EMA20** → Line appears, no visual jank.
4. ✅ **Toggle dark/light** → instant theme swap.
5. ✅ **Reload page** → Restores last range, theme, indicators.
6. ✅ **Live ticks** → Last bar updates without resetting viewport.

---

## Responsive Design

- **Desktop (1200px+)** – Full-width layout, side-by-side controls
- **Tablet (768px+)** – Stacked controls, auto chart height
- **Mobile (< 768px)** – Touch-friendly buttons, swipe pan, pinch zoom

Tailwind responsive utilities used throughout (`md:`, `flex-wrap`, etc.).

---

## Zustand Store Persistence

Stored in `localStorage` under key `capital-like-chart-state-v1`:

```json
{
  "symbol": "XAUUSD",
  "range": "30D",
  "resolution": "15m",
  "chartType": "candles",
  "theme": "dark",
  "volumeVisible": true,
  "indicators": {
    "ema20": false,
    "ema50": false,
    "ema200": false,
    "atr14": false
  },
  "visibleRange": {
    "from": 1641000000,
    "to": 1641086400
  }
}
```

Clear with: `localStorage.removeItem("capital-like-chart-state-v1")`

---

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+ (requires `idb` polyfill for IndexedDB)
- Mobile browsers (iOS Safari 14+, Chrome Mobile)

---

## License

MIT
