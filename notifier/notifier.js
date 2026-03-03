'use strict';

const { Client, LocalAuth } = require('whatsapp-web.js');
const express = require('express');
const qrcode = require('qrcode-terminal');

const PORT = process.env.PORT || 3099;
const WA_NUMBER = process.env.WA_NUMBER; // e.g. 962XXXXXXXXX (no +)

if (!WA_NUMBER) {
    console.error('[WhatsApp] ERROR: WA_NUMBER environment variable is required.');
    console.error('  Export it before starting: export WA_NUMBER=962XXXXXXXXX');
    process.exit(1);
}

const app = express();
app.use(express.json());

// ── Puppeteer errors that indicate a dead browser session ────────
const FATAL_PATTERNS = [
    'detached Frame',
    'detached frame',
    'Session closed',
    'Target closed',
    'Protocol error',
    'Page crashed',
    'Cannot find context',
    'Execution context was destroyed',
];

function isFatalPuppeteerError(msg) {
    return FATAL_PATTERNS.some(p => msg.includes(p));
}

// ── Client factory ───────────────────────────────────────────────
function makeClient() {
    return new Client({
        authStrategy: new LocalAuth({ dataPath: './.wwebjs_auth' }),
        puppeteer: {
            headless: true,
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-first-run',
                '--no-zygote',
                // NOTE: --single-process removed; it causes crashes on macOS/Linux
                '--disable-gpu',
                '--disable-features=site-per-process',  // prevents frame detaches
                '--disable-site-isolation-trials',
            ],
        },
    });
}

let client = makeClient();
let ready = false;
let qrPending = false;
let resetting = false;

// ── Session reset ────────────────────────────────────────────────
async function resetClient(reason) {
    if (resetting) return;
    resetting = true;
    ready = false;
    console.warn(`[WhatsApp] Resetting session — reason: ${reason}`);
    try { await client.destroy(); } catch (_) { /* ignore */ }
    setTimeout(() => {
        client = makeClient();
        attachEvents();
        client.initialize();
        resetting = false;
        console.log('[WhatsApp] Re-initialising client...');
    }, 6000);
}

// ── Attach WhatsApp events ────────────────────────────────────────
function attachEvents() {
    client.on('qr', (qr) => {
        qrPending = true;
        console.log('\n[WhatsApp] ── QR CODE ── Scan this in WhatsApp → Settings → Linked Devices:\n');
        qrcode.generate(qr, { small: true });
        console.log('\n[WhatsApp] Waiting for scan...\n');
    });

    client.on('authenticated', () => {
        qrPending = false;
        console.log('[WhatsApp] Authenticated successfully.');
    });

    client.on('ready', () => {
        ready = true;
        qrPending = false;
        console.log(`[WhatsApp] Client ready. Sending to ${WA_NUMBER}@c.us`);
    });

    client.on('disconnected', (reason) => {
        ready = false;
        resetClient(`disconnected event: ${reason}`);
    });

    client.on('auth_failure', (msg) => {
        console.error(`[WhatsApp] Auth failure: ${msg}`);
        ready = false;
        resetClient(`auth_failure: ${msg}`);
    });

    // Catch unhandled Puppeteer errors emitted on the client
    client.on('change_state', (state) => {
        console.log(`[WhatsApp] State → ${state}`);
    });
}

// ── Watchdog: every 60 s probe the Puppeteer page ────────────────
setInterval(async () => {
    if (!ready || resetting) return;
    try {
        await client.getState();  // throws if page/context is dead
    } catch (e) {
        console.warn(`[WhatsApp] Watchdog detected dead session: ${e.message}`);
        resetClient(`watchdog: ${e.message}`);
    }
}, 60_000);

// ── HTTP endpoints ───────────────────────────────────────────────

/**
 * POST /send
 * Body: { "message": "..." }
 */
app.post('/send', async (req, res) => {
    if (!ready) {
        const state = resetting ? 'resetting' : qrPending ? 'waiting_for_qr_scan' : 'initialising';
        return res.status(503).json({ error: 'Client not ready', state });
    }

    const { message } = req.body;
    if (!message || typeof message !== 'string' || !message.trim()) {
        return res.status(400).json({ error: 'Non-empty message string required' });
    }

    try {
        const chatId = `${WA_NUMBER}@c.us`;
        const result = await client.sendMessage(chatId, message.trim());
        console.log(`[WhatsApp] Sent (msgId=${result.id._serialized}): ${message.slice(0, 80)}`);
        res.json({ ok: true, msgId: result.id._serialized, timestamp: Date.now() });
    } catch (e) {
        console.error('[WhatsApp] Send error:', e.message);
        // If Puppeteer session is dead, recover automatically
        if (isFatalPuppeteerError(e.message)) {
            resetClient(`send error: ${e.message}`);
            return res.status(503).json({ error: 'Session lost — reconnecting', detail: e.message });
        }
        res.status(500).json({ error: e.message });
    }
});

/**
 * GET /health
 */
app.get('/health', (req, res) => {
    res.json({
        ready,
        qrPending,
        resetting,
        uptime: Math.floor(process.uptime()),
        target: `${WA_NUMBER}@c.us`,
    });
});

// ── Start ────────────────────────────────────────────────────────

app.listen(PORT, '127.0.0.1', () => {
    console.log(`[WhatsApp] Notifier listening on http://127.0.0.1:${PORT}`);
    console.log(`[WhatsApp] Target number: ${WA_NUMBER}`);
});

attachEvents();
client.initialize();

// Graceful shutdown
process.on('SIGINT', async () => {
    console.log('\n[WhatsApp] Shutting down...');
    try { await client.destroy(); } catch (_) { }
    process.exit(0);
});
