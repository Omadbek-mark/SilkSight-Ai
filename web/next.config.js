const { i18n } = require('./next-i18next.config');

/** @type {import('next').NextConfig} */
const nextConfig = {
	reactStrictMode: true,
	i18n,
};

// Keep `next build` from replacing files used by a running dev server (same as Hotelia-next).
module.exports = (phase) => ({
	...nextConfig,
	distDir: phase === 'phase-development-server' ? '.next-dev' : '.next',
});
