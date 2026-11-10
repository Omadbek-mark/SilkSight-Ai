module.exports = {
	i18n: {
		defaultLocale: 'en',
		locales: ['en', 'uz'],
		localeDetection: false,
	},
	fallbackLng: 'en',
	defaultNS: 'common',
	keySeparator: false,
	resources: {
		en: { common: require('./public/locales/en/common.json') },
		uz: { common: require('./public/locales/uz/common.json') },
	},
};
