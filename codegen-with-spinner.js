#!/usr/bin/env node
const { spawn } = require('child_process');

const url = process.argv[2] || 'https://playwright.dev/';
const args = ['npx', 'playwright', 'codegen', url];
const xvfbArgs = ['xvfb-run', ...args];

// Simple spinner
const spinnerFrames = ['|', '/', '-', '\\'];
let spinnerIndex = 0;
process.stdout.write('Launching Playwright codegen ');
const spinner = setInterval(() => {
  process.stdout.write('\rLaunching Playwright codegen ' + spinnerFrames[spinnerIndex++ % spinnerFrames.length]);
}, 100);

const child = spawn(xvfbArgs[0], xvfbArgs.slice(1), { stdio: 'inherit' });

child.on('exit', (code) => {
  clearInterval(spinner);
  process.stdout.write('\rPlaywright codegen exited with code ' + code + '\n');
  process.exit(code);
});
