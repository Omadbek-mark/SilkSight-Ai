import { Button, Typography } from '@mui/material';
import { useTranslation } from 'next-i18next';
import { useState } from 'react';
import { useLocale } from '../../hooks/useLocalized';
import { LandmarkDetail } from '../../types/landmark';

const PREVIEW_CHARS = 600;

/** Wikipedia lead text in the current locale, or English with a note; never translated. */
export default function WikipediaText({ landmark }: { landmark: LandmarkDetail }) {
	const { t } = useTranslation('common');
	const locale = useLocale();
	const [expanded, setExpanded] = useState(false);
	const article = landmark.summary[locale] ?? landmark.summary.en;

	if (!article) {
		return (
			<Typography color="text.secondary" className="wiki-text">
				{t('landmark.noText')}
			</Typography>
		);
	}
	const long = article.text.length > PREVIEW_CHARS;
	const shown = expanded || !long ? article.text : `${article.text.slice(0, PREVIEW_CHARS).replace(/\s+\S*$/, '')}…`;

	return (
		<div className="wiki-text" lang={article.language}>
			{article.language !== locale && (
				<Typography variant="body2" color="text.secondary" className="wiki-text__note">
					{t('landmark.englishOnly')}
				</Typography>
			)}
			{shown.split('\n\n').map((paragraph, index) => (
				<Typography key={index} paragraph>
					{paragraph}
				</Typography>
			))}
			{long && (
				<Button size="small" onClick={() => setExpanded(!expanded)}>
					{expanded ? t('landmark.showLess') : t('landmark.readMore')}
				</Button>
			)}
			<Typography variant="caption" color="text.secondary" component="p" className="wiki-text__source">
				<a href={article.url} target="_blank" rel="noopener noreferrer">
					{t('landmark.source', { title: article.title, license: article.license })}
				</a>
			</Typography>
		</div>
	);
}
