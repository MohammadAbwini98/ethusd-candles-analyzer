#!/usr/bin/env node
'use strict';

/**
 * Helper script to find WhatsApp group chat IDs
 * 
 * Usage:
 *   node wa_get_group_id.js
 * 
 * This will list all your WhatsApp group chats with their names and IDs.
 * Copy the ID for your "ETHUSD Analyzer" group and paste it in config.yaml.
 */

const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');

const client = new Client({
    authStrategy: new LocalAuth({ dataPath: './.wwebjs_auth' }),
    puppeteer: {
        headless: false,  // Show browser for debugging
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
        ],
    },
});

console.log('[WhatsApp] Initializing client...');
console.log('[WhatsApp] This will use your existing session from .wwebjs_auth\n');

client.on('qr', (qr) => {
    console.log('[WhatsApp] Scan this QR code with your phone:');
    qrcode.generate(qr, { small: true });
});

client.on('ready', async () => {
    console.log('\n[WhatsApp] ✓ Client ready!\n');
    console.log('═══════════════════════════════════════════════════════════');
    console.log('                   YOUR WHATSAPP GROUPS                     ');
    console.log('═══════════════════════════════════════════════════════════\n');

    try {
        const chats = await client.getChats();
        const groups = chats.filter(chat => chat.isGroup);

        if (groups.length === 0) {
            console.log('  No groups found in your WhatsApp account.\n');
        } else {
            groups.forEach((group, index) => {
                console.log(`${index + 1}. ${group.name}`);
                console.log(`   ID: ${group.id._serialized}`);
                console.log(`   Participants: ${group.participants.length}`);
                console.log('');
            });

            console.log('───────────────────────────────────────────────────────────');
            console.log('To use a group in config.yaml:');
            console.log('  1. Copy the ID of your "ETHUSD Analyzer" group above');
            console.log('  2. Update config.yaml:');
            console.log('     wa_number: "YOUR_GROUP_ID_HERE"');
            console.log('───────────────────────────────────────────────────────────\n');
        }

        await client.destroy();
        process.exit(0);
    } catch (err) {
        console.error('[Error]', err.message);
        process.exit(1);
    }
});

client.on('auth_failure', (msg) => {
    console.error('[WhatsApp] Authentication failure:', msg);
    process.exit(1);
});

client.initialize();
