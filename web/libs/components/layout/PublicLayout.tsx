import { Container, Stack, Typography } from '@mui/material';
import { useTranslation } from 'next-i18next';
import Head from 'next/head';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { ReactNode } from 'react';

const LOCALES = [
	{ code: 'en', label: 'EN' },
	{ code: 'uz', label: 'UZ' },
];

export default function PublicLayout({ children }: { children: ReactNode }) {
	const { t } = useTranslation('common');
	const router = useRouter();
	const links = [
		{ href: '/', label: t('nav.recognize') },
		{ href: '/landmark', label: t('nav.landmarks') },
	];

	return (
		<div className="layout">
			<Head>
				<title>SilkSight</title>
				<meta name="viewport" content="width=device-width, initial-scale=1" />
			</Head>
			<header className="layout__header">
				<Container maxWidth="lg">
					<div className="layout__bar">
						<Link href="/" className="layout__brand">
							Silk<span>Sight</span>
						</Link>
						<nav className="layout__nav">
							{links.map((link) => (
								<Link
									key={link.href}
									href={link.href}
									className={router.pathname === link.href ? 'is-active' : undefined}
								>
									{link.label}
								</Link>
							))}
						</nav>
						<Stack direction="row" spacing={0.5} aria-label={t('nav.language')} className="layout__locales">
							{LOCALES.map((locale) => (
								<Link
									key={locale.code}
									href={{ pathname: router.pathname, query: router.query }}
									locale={locale.code}
									className={router.locale === locale.code ? 'is-active' : undefined}
								>
									{locale.label}
								</Link>
							))}
						</Stack>
					</div>
				</Container>
			</header>
			<main className="layout__main">{children}</main>
			<footer className="layout__footer">
				<Container maxWidth="lg">
					<Typography variant="body2" color="text.secondary">
						{t('footer.credits')}
					</Typography>
				</Container>
			</footer>
		</div>
	);
}
