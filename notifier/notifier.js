'use strict';

const { Client, LocalAuth } = require('whatsapp-web.js');
const express = require('express');
const qrcode = require('qrcode-terminal');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3099;
const WA_NUMBER = process.env.WA_NUMBER; // e.g. 962XXXXXXXXX (no +) or group ID like 120363XXXXX@g.us

if (!WA_NUMBER) {
    console.error('[WhatsApp] ERROR: WA_NUMBER environment variable is required.');
    console.error('  For individual: export WA_NUMBER=962XXXXXXXXX');
    console.error('  For group: export WA_NUMBER=120363XXXXX@g.us');
    process.exit(1);
}

// ── Determine chat ID format (individual vs group) ──────────────
function getChatId(waNumber) {
    // If already has a suffix (@c.us or @g.us), use as-is
    if (waNumber.includes('@')) {
        return waNumber;
    }
    // Otherwise, treat as individual number and append @c.us
    return `${waNumber}@c.us`;
}

const CHAT_ID = getChatId(WA_NUMBER);
const IS_GROUP = CHAT_ID.endsWith('@g.us');

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
let shuttingDown = false;  // set true on SIGINT/SIGTERM; rejects new /send requests

// ── Pre-ready message queue ──────────────────────────────────────
// Messages arriving while the client is still initialising/resetting are
// stored here and flushed automatically the moment 'ready' fires.
const QUEUE_MAX = 50;
const pendingQueue = [];   // [{ message: string, ts: number }, ...]

async function flushPendingQueue() {
    if (pendingQueue.length === 0) return;
    console.log(`[WhatsApp] Flushing ${pendingQueue.length} queued message(s)...`);
    const items = pendingQueue.splice(0);
    for (const item of items) {
        try {
            const result = await client.sendMessage(CHAT_ID, item.message);
            console.log(`[WhatsApp] Flushed queued msg (msgId=${result.id._serialized}): ${item.message.slice(0, 60)}`);
        } catch (e) {
            console.error(`[WhatsApp] Failed to flush queued message: ${e.message}`);
            // Re-queue only if session is not fatally dead
            if (!isFatalPuppeteerError(e.message) && pendingQueue.length < QUEUE_MAX) {
                pendingQueue.unshift(item);
            }
        }
    }
}

// ── Session reset ────────────────────────────────────────────────
async function resetClient(reason) {
    if (resetting || shuttingDown) return;
    resetting = true;   // stays true until 'ready' fires OR initialize() fails
    ready = false;
    console.warn(`[WhatsApp] Resetting session — reason: ${reason}`);
    try { await client.destroy(); } catch (_) { /* ignore */ }
    setTimeout(() => {
        cleanBrowserLock();
        client = makeClient();
        attachEvents();
        console.log('[WhatsApp] Re-initialising client...');
        client.initialize().catch(err => {
            console.error(`[WhatsApp] initialize() failed during reset: ${err.message}`);
            resetting = false;   // unblock so the next attempt can proceed
            // Back off 12 s before trying again to let the OS fully release Chrome.
            if (!shuttingDown) setTimeout(() => resetClient(`re-init failed: ${err.message}`), 12000);
        });
        // NOTE: resetting is cleared in the 'ready' event handler, not here.
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
        resetting = false;   // clear here, not in resetClient(), so timing is correct
        qrPending = false;
        const targetType = IS_GROUP ? 'group' : 'individual';
        console.log(`[WhatsApp] Client ready. Sending to ${targetType}: ${CHAT_ID}`);
        // Deliver any messages that arrived before the client was ready.
        flushPendingQueue();
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
const watchdogInterval = setInterval(async () => {
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
    if (shuttingDown) {
        return res.status(503).json({ error: 'Server shutting down' });
    }
    if (!ready) {
        const state = resetting ? 'resetting' : qrPending ? 'waiting_for_qr_scan' : 'initialising';
        return res.status(503).json({ error: 'Client not ready', state });
    }

    const { message } = req.body;
    if (!message || typeof message !== 'string' || !message.trim()) {
        return res.status(400).json({ error: 'Non-empty message string required' });
    }

    try {
        const result = await client.sendMessage(CHAT_ID, message.trim());
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
        target: CHAT_ID,
        targetType: IS_GROUP ? 'group' : 'individual',
        queueLength: pendingQueue.length,
    });
});

// ── Start ────────────────────────────────────────────────────────

// Free port before binding (same fix as Python dashboard_server)
function freePort(port) {
    try {
        const pids = execSync(`lsof -ti:${port}`, { stdio: ['pipe', 'pipe', 'ignore'] })
            .toString().trim().split('\n').filter(Boolean);
        const myPid = String(process.pid);
        pids.forEach(pid => {
            if (pid !== myPid) {
                try { execSync(`kill -9 ${pid}`, { stdio: 'ignore' }); } catch (_) { }
            }
        });
        if (pids.length > 0) setTimeout(() => { }, 300); // brief pause for OS to release
    } catch (_) { }
}

// ── Clean up stale Chrome browser lock on startup ──────────────
function cleanBrowserLock() {
    const userDataDir = path.resolve('.wwebjs_auth', 'session');

    // 1. Remove lock files so Puppeteer can launch a fresh browser instance.
    for (const lock of ['SingletonLock', 'SingletonCookie', 'SingletonSocket']) {
        const f = path.join(userDataDir, lock);
        if (fs.existsSync(f)) {
            try { fs.unlinkSync(f); console.log(`[WhatsApp] Removed stale ${lock}`); } catch (_) { }
        }
    }

    // 2. Kill Chrome processes that have the profile directory open.
    //    Two strategies in sequence for maximum coverage:
    //    a) pkill -9 -f matching the profile path (most reliable on macOS/Linux)
    //    b) lsof fallback: find PIDs with the dir open and kill them
    let killed = 0;
    try {
        // pkill returns exit code 1 when no processes matched — that's fine.
        const pkillResult = execSync(
            `pkill -9 -f "${userDataDir}" 2>/dev/null; echo ok`,
            { stdio: ['pipe', 'pipe', 'ignore'] }
        ).toString().trim();
        // Check how many were killed
        try {
            const n = execSync(`pgrep -c -f "${userDataDir}" 2>/dev/null || echo 0`, { stdio: ['pipe', 'pipe', 'ignore'] }).toString().trim();
            killed = parseInt(n, 10) || 0;
        } catch (_) { }
    } catch (_) { }

    // lsof fallback
    try {
        const lsofPids = execSync(
            `lsof -t "${userDataDir}" 2>/dev/null || true`,
            { stdio: ['pipe', 'pipe', 'ignore'] }
        ).toString().trim().split('\n').filter(Boolean);
        const myPid = String(process.pid);
        lsofPids.forEach(pid => {
            if (pid && pid !== myPid) {
                try { execSync(`kill -9 ${pid}`, { stdio: 'ignore' }); killed++; } catch (_) { }
            }
        });
    } catch (_) { }

    if (killed > 0) {
        console.log(`[WhatsApp] Killed ${killed} stale Chrome process(es).`);
    }

    // 3. Spin-wait 1.5 s so the OS fully releases file handles before
    //    Puppeteer tries to open the same profile directory.
    const until = Date.now() + 1500;
    while (Date.now() < until) { /* spin-wait */ }
}

freePort(PORT);

const server = app.listen(PORT, '127.0.0.1', () => {
    console.log(`[WhatsApp] Notifier listening on http://127.0.0.1:${PORT}`);
    const targetType = IS_GROUP ? 'Group' : 'Individual';
    console.log(`[WhatsApp] Target (${targetType}): ${CHAT_ID}`);
});

server.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        console.log(`[WhatsApp] Port ${PORT} in use — retrying after kill...`);
        freePort(PORT);
        setTimeout(() => {
            server.close();
            app.listen(PORT, '127.0.0.1', () => {
                console.log(`[WhatsApp] Notifier listening on http://127.0.0.1:${PORT} (retry)`);
            });
        }, 800);
    } else {
        console.error('[WhatsApp] Server error:', err);
    }
});

// Catch unhandled promise rejections (e.g. Puppeteer launch failures)
process.on('unhandledRejection', (reason) => {
    const msg = reason instanceof Error ? reason.message : String(reason);
    console.error(`[WhatsApp] Unhandled rejection: ${msg}`);
    if (!resetting) {
        resetClient(`unhandledRejection: ${msg}`);
    }
});

attachEvents();
cleanBrowserLock();
client.initialize().catch(err => {
    console.error(`[WhatsApp] initialize() failed: ${err.message}`);
    resetClient(`initialize failed: ${err.message}`);
});

// Graceful shutdown
async function shutdown(sig) {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`\n[WhatsApp] ${sig} received — graceful shutdown...`);

    // Hard timeout: if anything hangs for more than 8 s, force-exit.
    const hardTimer = setTimeout(() => {
        console.error('[WhatsApp] Shutdown hard timeout — forcing exit (code 1).');
        process.exit(1);
    }, 8000);
    hardTimer.unref(); // don't prevent exit if everything closes cleanly

    // 1. Stop watchdog so it can't trigger resetClient during teardown.
    clearInterval(watchdogInterval);

    // 2. Stop accepting new HTTP connections; finish in-flight requests.
    server.close(() => console.log('[WhatsApp] HTTP server closed.'));

    // 3. Destroy WhatsApp / Puppeteer session.
    try {
        await client.destroy();
        console.log('[WhatsApp] Client destroyed.');
    } catch (_) { /* ignore — browser may already be gone */ }

    // 4. Optional GC (only available when Node is started with --expose-gc).
    if (typeof global.gc === 'function') {
        global.gc();
        console.log('[WhatsApp] GC collected.');
    }

    console.log('[WhatsApp] Shutdown complete.');
    clearTimeout(hardTimer);
    process.exit(0);
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGHUP', () => shutdown('SIGHUP'));   // terminal close / daemon restart
