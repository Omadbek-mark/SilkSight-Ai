import { useRouter } from 'next/router';
import { LocalizedName, Locale } from '../types/landmark';

export function useLocale(): Locale {
	return useRouter().locale === 'uz' ? 'uz' : 'en';
}

/** Landmark name in the current locale; the English name when no Uzbek one exists. */
export function useLocalizedName(): (name: LocalizedName) => string {
	const locale = useLocale();
	return (name) => (locale === 'uz' && name.uz) || name.en;
}
